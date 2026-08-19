"""
Margin model tests.

Every expected number here is derived from the published rule (Reg-T / CBOE
minimums, or full collateralization for cash accounts) and compared against the
engine in exact microdollars, never against a value read back out of the engine.
"""
from __future__ import annotations

import itertools

import pytest

import obt_engine as E
from tests.conftest import make_contract

SYMBOL = "TEST"
SPOT = 100.0
SHARES_PER_CONTRACT = 100

ALL_MODELS = [
    E.MarginModel.CASH_ACCOUNT,
    E.MarginModel.REG_T,
    E.MarginModel.ROBINHOOD,
    E.MarginModel.PORTFOLIO_APPROX,
]


def margin(model, contracts, holdings, *, spot=SPOT, marks=None, shares=None):
    return E.evaluate_margin(
        model, contracts, holdings, {SYMBOL: spot}, marks or {}, shares or {}
    )


def micros(dollars: float) -> int:
    return round(dollars * 1_000_000)


def stock_charge(shares: int, *, spot: float = SPOT, fraction: float = 0.50) -> float:
    """
    Reg-T margin on a stock holding.

    A covered call requires margin on the STOCK and none on the call, so a
    covered-call requirement is this number rather than zero.
    """
    return fraction * spot * shares


def reg_t_naked_dollars(
    *, strike, spot, is_call, premium, shares=SHARES_PER_CONTRACT, contracts=1
):
    """The 20%/10% CBOE minimum for one uncovered short option, per the rule text."""
    out_of_the_money = max(0.0, strike - spot) if is_call else max(0.0, spot - strike)
    floor = 0.10 * spot if is_call else 0.10 * strike
    per_share = max(0.20 * spot - out_of_the_money, floor, 0.0) + premium
    return per_share * shares * contracts


def only(pairings):
    assert len(pairings) == 1, pairings
    return pairings[0]


class TestLongOptions:
    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_long_only_position_adds_no_requirement(self, model):
        """A long option is paid for in cash at entry, so it collateralizes itself."""
        call = make_contract(1, strike=100.0, expiry_day=30)
        put = make_contract(2, strike=95.0, expiry_day=30, is_call=False)

        result = margin(model, [call, put], [(1, 5), (2, 3)], marks={1: 4.0, 2: 2.0})

        assert result.requirement_micros == 0
        assert not result.disallowed
        assert result.pairings == []


class TestPoorMansCoveredCall:
    """Long call below the short strike and expiring later caps the loss at the debit."""

    COMBOS = [
        (50.0, 365, 100.0, 30),
        (90.0, 180, 95.0, 60),
        (99.0, 45, 100.0, 44),
        (100.0, 60, 100.0, 30),
        (80.0, 720, 130.0, 21),
    ]

    @pytest.mark.parametrize("long_strike,long_expiry,short_strike,short_expiry", COMBOS)
    @pytest.mark.parametrize(
        "model",
        [E.MarginModel.CASH_ACCOUNT, E.MarginModel.REG_T, E.MarginModel.ROBINHOOD],
    )
    def test_pmcc_requires_no_margin_beyond_the_debit(
        self, model, long_strike, long_expiry, short_strike, short_expiry
    ):
        leaps = make_contract(1, strike=long_strike, expiry_day=long_expiry)
        short = make_contract(2, strike=short_strike, expiry_day=short_expiry)

        result = margin(model, [leaps, short], [(1, 1), (2, -1)], marks={2: 1.25})

        assert result.requirement_micros == 0
        assert not result.disallowed

    def test_pmcc_pairs_the_short_against_the_leaps_rather_than_leaving_it_naked(self):
        leaps = make_contract(1, strike=80.0, expiry_day=400)
        short = make_contract(2, strike=110.0, expiry_day=30)

        pairing = only(
            margin(E.MarginModel.REG_T, [leaps, short], [(1, 1), (2, -1)]).pairings
        )

        assert pairing.short_leg == 2
        assert pairing.long_leg == 1
        assert not pairing.naked
        assert pairing.requirement == 0.0

    def test_pmcc_scales_pairing_to_every_short_contract(self):
        leaps = make_contract(1, strike=80.0, expiry_day=400)
        short = make_contract(2, strike=110.0, expiry_day=30)

        result = margin(E.MarginModel.ROBINHOOD, [leaps, short], [(1, 3), (2, -3)])

        assert result.requirement_micros == 0
        assert only(result.pairings).contracts == 3


class TestVerticalSpreads:
    @pytest.mark.parametrize("width", [1.0, 5.0, 10.0])
    @pytest.mark.parametrize("contracts", [1, 2])
    def test_bear_call_spread_requires_exactly_the_strike_width(self, width, contracts):
        short = make_contract(1, strike=100.0, expiry_day=30)
        long = make_contract(2, strike=100.0 + width, expiry_day=30)

        result = margin(
            E.MarginModel.REG_T,
            [short, long],
            [(1, -contracts), (2, contracts)],
            marks={1: 3.0, 2: 1.0},
        )

        assert result.requirement_micros == micros(
            width * SHARES_PER_CONTRACT * contracts
        )
        assert not result.disallowed
        assert only(result.pairings).contracts == contracts

    @pytest.mark.parametrize("width", [1.0, 5.0, 10.0])
    @pytest.mark.parametrize("contracts", [1, 2])
    def test_bull_put_spread_requires_exactly_the_strike_width(self, width, contracts):
        short = make_contract(1, strike=100.0, expiry_day=30, is_call=False)
        long = make_contract(2, strike=100.0 - width, expiry_day=30, is_call=False)

        result = margin(
            E.MarginModel.REG_T,
            [short, long],
            [(1, -contracts), (2, contracts)],
            marks={1: 3.0, 2: 1.0},
        )

        assert result.requirement_micros == micros(
            width * SHARES_PER_CONTRACT * contracts
        )
        assert not result.disallowed

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_spread_width_is_charged_identically_by_every_model(self, model):
        short = make_contract(1, strike=100.0, expiry_day=30)
        long = make_contract(2, strike=105.0, expiry_day=30)

        result = margin(model, [short, long], [(1, -1), (2, 1)])

        assert result.requirement_micros == micros(500.0)
        assert not result.disallowed


class TestExpiryTrap:
    """
    A long leg that expires before the short does not cover it.

    The position is naked from the long's expiry to the short's, so treating it
    as a spread understates the risk arbitrarily. The previous engine accepted
    this silently; these are its regression tests.
    """

    @staticmethod
    def legs():
        long_expiring_first = make_contract(1, strike=90.0, expiry_day=20)
        short = make_contract(2, strike=110.0, expiry_day=30)
        return [long_expiring_first, short]

    def test_short_call_outliving_its_long_is_marked_naked(self):
        pairing = only(
            margin(E.MarginModel.REG_T, self.legs(), [(1, 1), (2, -1)]).pairings
        )

        assert pairing.naked
        assert pairing.long_leg == 0

    def test_robinhood_refuses_a_short_call_outliving_its_long(self):
        result = margin(E.MarginModel.ROBINHOOD, self.legs(), [(1, 1), (2, -1)])

        assert result.disallowed
        assert "not permitted" in result.disallowed_reason

    def test_reg_t_charges_the_naked_amount_for_a_short_call_outliving_its_long(self):
        result = margin(
            E.MarginModel.REG_T, self.legs(), [(1, 1), (2, -1)], marks={2: 1.5}
        )

        expected = reg_t_naked_dollars(
            strike=110.0, spot=SPOT, is_call=True, premium=1.5
        )
        assert result.requirement_micros == micros(expected)
        assert result.requirement_micros > 0

    def test_short_put_outliving_its_long_is_collateralized_at_the_full_strike(self):
        long_expiring_first = make_contract(1, strike=90.0, expiry_day=20, is_call=False)
        short = make_contract(2, strike=95.0, expiry_day=30, is_call=False)

        result = margin(
            E.MarginModel.ROBINHOOD, [long_expiring_first, short], [(1, 1), (2, -1)]
        )

        assert result.requirement_micros == micros(95.0 * SHARES_PER_CONTRACT)
        assert only(result.pairings).naked

    def test_a_long_expiring_on_the_short_expiry_still_covers_it(self):
        long = make_contract(1, strike=90.0, expiry_day=30)
        short = make_contract(2, strike=110.0, expiry_day=30)

        result = margin(E.MarginModel.ROBINHOOD, [long, short], [(1, 1), (2, -1)])

        assert result.requirement_micros == 0
        assert not only(result.pairings).naked


class TestCoveredCall:
    @pytest.mark.parametrize(
        "model", [E.MarginModel.CASH_ACCOUNT, E.MarginModel.REG_T, E.MarginModel.ROBINHOOD]
    )
    def test_shares_cover_the_short_call_so_only_the_stock_is_charged(self, model):
        """
        Reg-T charges margin on the stock and nothing on the call. A cash account
        pays for the stock in full.
        """
        short = make_contract(1, strike=105.0, expiry_day=30)

        result = margin(model, [short], [(1, -1)], shares={SYMBOL: 100}, marks={1: 2.0})

        fraction = 1.00 if model == E.MarginModel.CASH_ACCOUNT else 0.50
        assert result.requirement_micros == micros(stock_charge(100, fraction=fraction))
        assert not result.disallowed
        assert only(result.pairings).covered_by_equity

    def test_two_hundred_shares_cover_two_short_calls(self):
        short = make_contract(1, strike=105.0, expiry_day=30)

        result = margin(
            E.MarginModel.ROBINHOOD, [short], [(1, -2)], shares={SYMBOL: 200}
        )

        assert result.requirement_micros == micros(stock_charge(200))
        assert not result.disallowed
        pairing = only(result.pairings)
        assert pairing.covered_by_equity
        assert pairing.contracts == 2

    def test_one_hundred_shares_cover_only_one_of_two_short_calls(self):
        """
        Coverage is apportioned per contract, so the holding covers one and
        leaves the other uncovered rather than covering neither.
        """
        short = make_contract(1, strike=105.0, expiry_day=30)

        result = margin(
            E.MarginModel.ROBINHOOD, [short], [(1, -2)], shares={SYMBOL: 100}
        )

        assert result.disallowed
        covered = [p for p in result.pairings if p.covered_by_equity]
        uncovered = [p for p in result.pairings if not p.covered_by_equity]
        assert sum(p.contracts for p in covered) == 1
        assert sum(p.contracts for p in uncovered) == 1

    def test_reg_t_charges_only_the_uncovered_contract_when_shares_cover_one_of_two(self):
        short = make_contract(1, strike=105.0, expiry_day=30)

        result = margin(
            E.MarginModel.REG_T,
            [short],
            [(1, -2)],
            shares={SYMBOL: 100},
            marks={1: 2.0},
        )

        expected = reg_t_naked_dollars(
            strike=105.0, spot=SPOT, is_call=True, premium=2.0
        ) + stock_charge(100)
        assert result.requirement_micros == micros(expected)

    def test_shares_are_consumed_by_one_short_call_and_not_reused_by_another(self):
        near_the_money = make_contract(1, strike=105.0, expiry_day=30)
        further_out = make_contract(2, strike=110.0, expiry_day=30)

        result = margin(
            E.MarginModel.REG_T,
            [near_the_money, further_out],
            [(1, -1), (2, -1)],
            shares={SYMBOL: 100},
            marks={1: 2.0, 2: 1.0},
        )

        expected = reg_t_naked_dollars(
            strike=110.0, spot=SPOT, is_call=True, premium=1.0
        ) + stock_charge(100)
        assert result.requirement_micros == micros(expected)
        covered = [p for p in result.pairings if p.covered_by_equity]
        assert [p.short_leg for p in covered] == [1]

    def test_shares_do_not_offset_a_short_put(self):
        """Long stock plus a short put is a doubled-up bull, not a covered position."""
        short_put = make_contract(1, strike=95.0, expiry_day=30, is_call=False)

        result = margin(
            E.MarginModel.ROBINHOOD, [short_put], [(1, -1)], shares={SYMBOL: 100}
        )

        # The put is secured at its full strike, and the long stock is margined
        # in its own right rather than offsetting it.
        assert result.requirement_micros == micros(
            95.0 * SHARES_PER_CONTRACT + stock_charge(100)
        )
        assert not only(result.pairings).covered_by_equity


class TestCashSecuredPut:
    @pytest.mark.parametrize(
        "model", [E.MarginModel.ROBINHOOD, E.MarginModel.CASH_ACCOUNT]
    )
    @pytest.mark.parametrize("strike,contracts", [(95.0, 1), (95.0, 2), (12.5, 3)])
    def test_uncovered_short_put_is_secured_at_the_full_strike(
        self, model, strike, contracts
    ):
        short = make_contract(1, strike=strike, expiry_day=30, is_call=False)

        result = margin(model, [short], [(1, -contracts)], marks={1: 3.0})

        assert result.requirement_micros == micros(
            strike * SHARES_PER_CONTRACT * contracts
        )
        assert not result.disallowed

    def test_cash_secured_put_requirement_ignores_the_premium_received(self):
        short = make_contract(1, strike=95.0, expiry_day=30, is_call=False)

        with_premium = margin(E.MarginModel.ROBINHOOD, [short], [(1, -1)], marks={1: 9.0})
        without_premium = margin(E.MarginModel.ROBINHOOD, [short], [(1, -1)])

        assert with_premium.requirement_micros == without_premium.requirement_micros


class TestRegTNakedFormula:
    """max(20% * spot - out_of_the_money, floor) + premium, times the deliverable."""

    @pytest.mark.parametrize(
        "strike,is_call,premium",
        [
            (100.0, True, 3.25),
            (100.0, False, 3.25),
            (200.0, True, 0.05),
            (60.0, False, 0.40),
            (50.0, True, 50.50),
            (150.0, False, 50.25),
        ],
        ids=[
            "atm_call",
            "atm_put",
            "deep_otm_call_floor_binds",
            "deep_otm_put_floor_binds",
            "deep_itm_call",
            "deep_itm_put",
        ],
    )
    def test_uncovered_short_matches_the_published_minimum(self, strike, is_call, premium):
        short = make_contract(1, strike=strike, expiry_day=30, is_call=is_call)

        result = margin(
            E.MarginModel.REG_T, [short], [(1, -1)], marks={1: premium}
        )

        expected = reg_t_naked_dollars(
            strike=strike, spot=SPOT, is_call=is_call, premium=premium
        )
        assert result.requirement_micros == micros(expected)

    def test_deep_otm_call_requirement_is_the_ten_percent_floor_plus_premium(self):
        short = make_contract(1, strike=200.0, expiry_day=30)

        result = margin(E.MarginModel.REG_T, [short], [(1, -1)], marks={1: 0.05})

        assert result.requirement_micros == micros((0.10 * SPOT + 0.05) * 100)

    def test_deep_otm_put_floor_is_ten_percent_of_strike_not_of_spot(self):
        short = make_contract(1, strike=60.0, expiry_day=30, is_call=False)

        result = margin(E.MarginModel.REG_T, [short], [(1, -1)], marks={1: 0.40})

        assert result.requirement_micros == micros((0.10 * 60.0 + 0.40) * 100)

    @pytest.mark.parametrize("contracts", [1, 3, 10])
    def test_naked_requirement_is_linear_in_contract_count(self, contracts):
        short = make_contract(1, strike=100.0, expiry_day=30)

        result = margin(
            E.MarginModel.REG_T, [short], [(1, -contracts)], marks={1: 3.0}
        )

        per_contract = reg_t_naked_dollars(
            strike=100.0, spot=SPOT, is_call=True, premium=3.0
        )
        assert result.requirement_micros == micros(per_contract * contracts)

    def test_portfolio_approx_falls_back_to_reg_t_numbers(self):
        short = make_contract(1, strike=100.0, expiry_day=30)

        approx = margin(E.MarginModel.PORTFOLIO_APPROX, [short], [(1, -1)], marks={1: 3.0})
        reg_t = margin(E.MarginModel.REG_T, [short], [(1, -1)], marks={1: 3.0})

        assert approx.requirement_micros == reg_t.requirement_micros


class TestBrokerPermissions:
    def test_uncovered_short_call_is_refused_at_robinhood(self):
        short = make_contract(1, strike=100.0, expiry_day=30)

        result = margin(E.MarginModel.ROBINHOOD, [short], [(1, -1)], marks={1: 3.0})

        assert result.disallowed
        assert result.disallowed_reason
        assert "not permitted" in result.disallowed_reason

    def test_uncovered_short_call_is_refused_in_a_cash_account(self):
        short = make_contract(1, strike=100.0, expiry_day=30)

        result = margin(E.MarginModel.CASH_ACCOUNT, [short], [(1, -1)])

        assert result.disallowed
        assert result.disallowed_reason

    def test_uncovered_short_call_is_allowed_under_reg_t(self):
        short = make_contract(1, strike=100.0, expiry_day=30)

        result = margin(E.MarginModel.REG_T, [short], [(1, -1)], marks={1: 3.0})

        assert not result.disallowed
        assert result.disallowed_reason == ""
        assert result.requirement > 0.0

    def test_uncovered_short_put_is_allowed_at_robinhood(self):
        short = make_contract(1, strike=95.0, expiry_day=30, is_call=False)

        result = margin(E.MarginModel.ROBINHOOD, [short], [(1, -1)])

        assert not result.disallowed


class TestIronCondor:
    PUT_WIDTH = 5.0
    CALL_WIDTH = 10.0

    @staticmethod
    def legs():
        return [
            make_contract(1, strike=90.0, expiry_day=30, is_call=False),
            make_contract(2, strike=95.0, expiry_day=30, is_call=False),
            make_contract(3, strike=105.0, expiry_day=30),
            make_contract(4, strike=115.0, expiry_day=30),
        ]

    HOLDINGS = [(1, 1), (2, -1), (3, -1), (4, 1)]

    def test_iron_condor_is_charged_only_the_wider_side(self):
        """
        Only one wing of a condor can finish in the money, so the max-loss
        netting in FINRA 4210(f)(2)(H)(i) charges the greater width alone.
        Summing both widths would double-charge a risk that cannot occur.
        """
        result = margin(E.MarginModel.REG_T, self.legs(), self.HOLDINGS)

        assert result.requirement_micros == micros(
            max(self.PUT_WIDTH, self.CALL_WIDTH) * SHARES_PER_CONTRACT
        )
        assert result.requirement_micros < micros(
            (self.PUT_WIDTH + self.CALL_WIDTH) * SHARES_PER_CONTRACT
        )

    def test_iron_condor_pairs_each_short_with_its_own_wing(self):
        result = margin(E.MarginModel.REG_T, self.legs(), self.HOLDINGS)

        pairs = {p.short_leg: (p.long_leg, p.requirement) for p in result.pairings}
        assert pairs == {
            2: (1, self.PUT_WIDTH * SHARES_PER_CONTRACT),
            3: (4, self.CALL_WIDTH * SHARES_PER_CONTRACT),
        }
        assert not any(p.naked for p in result.pairings)

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_iron_condor_is_permitted_by_every_model(self, model):
        result = margin(model, self.legs(), self.HOLDINGS)

        assert not result.disallowed


class TestRatioAndPartialPairing:
    @staticmethod
    def call_ratio():
        long = make_contract(1, strike=100.0, expiry_day=30)
        short = make_contract(2, strike=110.0, expiry_day=30)
        return [long, short]

    def test_reg_t_charges_the_unpaired_leg_of_a_call_ratio_spread(self):
        result = margin(
            E.MarginModel.REG_T, self.call_ratio(), [(1, 1), (2, -2)], marks={2: 1.5}
        )

        expected = reg_t_naked_dollars(
            strike=110.0, spot=SPOT, is_call=True, premium=1.5
        )
        assert result.requirement_micros == micros(expected)
        paired = [p for p in result.pairings if not p.naked]
        naked = [p for p in result.pairings if p.naked]
        assert [p.contracts for p in paired] == [1]
        assert [p.contracts for p in naked] == [1]

    @pytest.mark.parametrize(
        "model", [E.MarginModel.ROBINHOOD, E.MarginModel.CASH_ACCOUNT]
    )
    def test_call_ratio_spread_is_refused_where_uncovered_calls_are(self, model):
        result = margin(model, self.call_ratio(), [(1, 1), (2, -2)])

        assert result.disallowed
        assert any(p.naked for p in result.pairings)

    def test_put_ratio_spread_secures_the_unpaired_short_put_at_its_strike(self):
        long = make_contract(1, strike=100.0, expiry_day=30, is_call=False)
        short = make_contract(2, strike=95.0, expiry_day=30, is_call=False)

        result = margin(E.MarginModel.ROBINHOOD, [long, short], [(1, 1), (2, -2)])

        assert result.requirement_micros == micros(95.0 * SHARES_PER_CONTRACT)
        assert not result.disallowed

    def test_one_long_call_against_three_shorts_leaves_two_uncovered(self):
        result = margin(
            E.MarginModel.REG_T, self.call_ratio(), [(1, 1), (2, -3)], marks={2: 1.5}
        )

        per_contract = reg_t_naked_dollars(
            strike=110.0, spot=SPOT, is_call=True, premium=1.5
        )
        assert result.requirement_micros == micros(per_contract * 2)
        naked = [p for p in result.pairings if p.naked]
        assert [p.contracts for p in naked] == [2]

    def test_two_long_calls_against_three_shorts_leave_one_uncovered(self):
        result = margin(
            E.MarginModel.REG_T, self.call_ratio(), [(1, 2), (2, -3)], marks={2: 1.5}
        )

        per_contract = reg_t_naked_dollars(
            strike=110.0, spot=SPOT, is_call=True, premium=1.5
        )
        assert result.requirement_micros == micros(per_contract)

    def test_more_longs_than_shorts_leaves_nothing_uncovered(self):
        result = margin(
            E.MarginModel.ROBINHOOD, self.call_ratio(), [(1, 5), (2, -2)]
        )

        assert result.requirement_micros == 0
        assert not any(p.naked for p in result.pairings)


class TestAdjustedDeliverable:
    """Post-split contracts still quote on 100 but deliver a different share count."""

    def test_spread_width_scales_with_the_real_deliverable(self):
        short = make_contract(1, strike=100.0, expiry_day=30, deliverable_shares=400)
        long = make_contract(2, strike=105.0, expiry_day=30, deliverable_shares=400)

        result = margin(E.MarginModel.REG_T, [short, long], [(1, -1), (2, 1)])

        assert result.requirement_micros == micros(5.0 * 400)

    def test_cash_secured_put_scales_with_the_real_deliverable(self):
        short = make_contract(
            1, strike=95.0, expiry_day=30, is_call=False, deliverable_shares=400
        )

        result = margin(E.MarginModel.ROBINHOOD, [short], [(1, -1)])

        assert result.requirement_micros == micros(95.0 * 400)

    def test_reg_t_naked_requirement_scales_with_the_real_deliverable(self):
        short = make_contract(1, strike=100.0, expiry_day=30, deliverable_shares=400)

        result = margin(E.MarginModel.REG_T, [short], [(1, -1)], marks={1: 3.0})

        expected = reg_t_naked_dollars(
            strike=100.0, spot=SPOT, is_call=True, premium=3.0, shares=400
        )
        assert result.requirement_micros == micros(expected)

    def test_covering_an_adjusted_short_call_needs_the_full_deliverable(self):
        short = make_contract(1, strike=105.0, expiry_day=30, deliverable_shares=400)

        covered = margin(
            E.MarginModel.ROBINHOOD, [short], [(1, -1)], shares={SYMBOL: 400}
        )
        under_covered = margin(
            E.MarginModel.ROBINHOOD, [short], [(1, -1)], shares={SYMBOL: 100}
        )

        # Covering a 400-share deliverable needs 400 shares, and those shares are
        # themselves margined; the call adds nothing.
        assert covered.requirement_micros == micros(stock_charge(400))
        assert only(covered.pairings).covered_by_equity
        assert under_covered.disallowed

    def test_quote_multiplier_alone_does_not_set_the_requirement(self):
        standard = make_contract(1, strike=95.0, expiry_day=30, is_call=False)
        adjusted = make_contract(
            2, strike=95.0, expiry_day=30, is_call=False, deliverable_shares=400
        )

        standard_result = margin(E.MarginModel.ROBINHOOD, [standard], [(1, -1)])
        adjusted_result = margin(E.MarginModel.ROBINHOOD, [adjusted], [(2, -1)])

        assert adjusted.quote_multiplier == standard.quote_multiplier == 100
        assert adjusted_result.requirement_micros == 4 * standard_result.requirement_micros


class TestPairingPreference:
    @pytest.mark.parametrize("order", [(1, 2), (2, 1)])
    def test_short_call_pairs_with_the_lower_strike_long(self, order):
        """Among eligible longs the cheapest pairing wins, regardless of input order."""
        short = make_contract(3, strike=100.0, expiry_day=30)
        near = make_contract(1, strike=105.0, expiry_day=30)
        far = make_contract(2, strike=110.0, expiry_day=30)
        by_id = {1: near, 2: far}

        result = margin(
            E.MarginModel.REG_T,
            [short] + [by_id[i] for i in order],
            [(3, -1), (order[0], 1), (order[1], 1)],
        )

        pairing = only(result.pairings)
        assert pairing.long_leg == 1
        assert result.requirement_micros == micros(5.0 * SHARES_PER_CONTRACT)

    @pytest.mark.parametrize("order", [(1, 2), (2, 1)])
    def test_short_put_pairs_with_the_higher_strike_long(self, order):
        short = make_contract(3, strike=100.0, expiry_day=30, is_call=False)
        near = make_contract(1, strike=95.0, expiry_day=30, is_call=False)
        far = make_contract(2, strike=90.0, expiry_day=30, is_call=False)
        by_id = {1: near, 2: far}

        result = margin(
            E.MarginModel.REG_T,
            [short] + [by_id[i] for i in order],
            [(3, -1), (order[0], 1), (order[1], 1)],
        )

        pairing = only(result.pairings)
        assert pairing.long_leg == 1
        assert result.requirement_micros == micros(5.0 * SHARES_PER_CONTRACT)

    def test_a_zero_residual_long_is_preferred_over_a_wider_one(self):
        short = make_contract(3, strike=100.0, expiry_day=30)
        protective = make_contract(1, strike=90.0, expiry_day=60)
        wider = make_contract(2, strike=120.0, expiry_day=60)

        result = margin(
            E.MarginModel.REG_T, [short, protective, wider], [(3, -1), (1, 1), (2, 1)]
        )

        assert only(result.pairings).long_leg == 1
        assert result.requirement_micros == 0

    def test_a_scarce_long_dated_long_goes_to_the_nearest_dated_short(self):
        near_short = make_contract(1, strike=100.0, expiry_day=30)
        far_short = make_contract(2, strike=100.0, expiry_day=60)
        long = make_contract(3, strike=100.0, expiry_day=60)

        result = margin(
            E.MarginModel.REG_T,
            [near_short, far_short, long],
            [(1, -1), (2, -1), (3, 1)],
            marks={1: 3.0, 2: 4.0},
        )

        paired = [p for p in result.pairings if not p.naked]
        assert [p.short_leg for p in paired] == [1]


class TestEdgeInputs:
    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_empty_holdings_require_nothing(self, model):
        result = margin(model, [], [])

        assert result.requirement_micros == 0
        assert not result.disallowed
        assert result.pairings == []

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_a_flat_position_is_ignored(self, model):
        short = make_contract(1, strike=100.0, expiry_day=30)

        result = margin(model, [short], [(1, 0)])

        assert result.requirement_micros == 0
        assert result.pairings == []

    def test_a_holding_in_an_unknown_contract_is_ignored(self):
        known = make_contract(1, strike=100.0, expiry_day=30)

        result = margin(E.MarginModel.REG_T, [known], [(1, 1), (999, -1)])

        assert result.requirement_micros == 0
        assert result.pairings == []

    def test_a_zero_underlying_price_yields_no_call_requirement(self):
        """Both the 20% term and the 10% call floor are proportional to spot."""
        short = make_contract(1, strike=100.0, expiry_day=30)

        result = margin(E.MarginModel.REG_T, [short], [(1, -1)], spot=0.0)

        assert result.requirement_micros == 0
        assert not result.disallowed

    def test_a_zero_underlying_price_still_charges_the_put_strike_floor(self):
        short = make_contract(1, strike=100.0, expiry_day=30, is_call=False)

        result = margin(E.MarginModel.REG_T, [short], [(1, -1)], spot=0.0)

        assert result.requirement_micros == micros(0.10 * 100.0 * SHARES_PER_CONTRACT)

    def test_a_missing_underlying_price_is_treated_as_zero_rather_than_crashing(self):
        short = make_contract(1, strike=100.0, expiry_day=30, is_call=False)

        result = E.evaluate_margin(E.MarginModel.REG_T, [short], [(1, -1)], {})

        assert result.requirement_micros == micros(0.10 * 100.0 * SHARES_PER_CONTRACT)

    def test_a_zero_strike_short_put_requires_nothing(self):
        short = make_contract(1, strike=0.0, expiry_day=30, is_call=False)

        result = margin(E.MarginModel.ROBINHOOD, [short], [(1, -1)])

        assert result.requirement_micros == 0

    def test_positions_in_separate_underlyings_are_summed(self):
        first = make_contract(1, strike=95.0, expiry_day=30, is_call=False)
        second = make_contract(
            2, strike=50.0, expiry_day=30, is_call=False, underlying="OTHER"
        )

        result = E.evaluate_margin(
            E.MarginModel.ROBINHOOD,
            [first, second],
            [(1, -1), (2, -1)],
            {SYMBOL: SPOT, "OTHER": 55.0},
        )

        assert result.requirement_micros == micros((95.0 + 50.0) * SHARES_PER_CONTRACT)
        assert len(result.pairings) == 2


class TestMixedDeliverablePairing:
    """
    Spread treatment requires the long and short sides to carry equal aggregate
    underlying value, not merely equal contract counts. A split leaves contracts
    with different deliverables on one underlying, which is exactly where a
    count-based pairing goes wrong.
    """

    LONG = 1
    SHORT = 2

    def _legs(self, long_shares, short_shares):
        return [
            make_contract(self.LONG, strike=100.0, expiry_day=400,
                          deliverable_shares=long_shares),
            make_contract(self.SHORT, strike=110.0, expiry_day=30,
                          deliverable_shares=short_shares),
        ]

    HOLDINGS = [(LONG, 1), (SHORT, -1)]

    def test_matched_deliverables_pair_to_zero(self):
        result = margin(E.MarginModel.REG_T, self._legs(100, 100), self.HOLDINGS,
                        marks={self.SHORT: 5.0})
        assert result.requirement_micros == 0
        assert not only(result.pairings).naked

    def test_short_delivering_more_shares_is_not_covered(self):
        """
        A 100-share long against a 400-share short leaves 300 shares of naked
        exposure whose loss is unbounded. Because the payoff slopes then fail to
        cancel, max-loss netting evaluated at the strikes reports a net gain and
        would charge nothing at all.
        """
        result = margin(E.MarginModel.REG_T, self._legs(100, 400), self.HOLDINGS,
                        marks={self.SHORT: 5.0})
        assert result.requirement_micros > 0
        assert only(result.pairings).naked

    def test_short_delivering_fewer_shares_is_also_not_covered(self):
        result = margin(E.MarginModel.REG_T, self._legs(400, 100), self.HOLDINGS,
                        marks={self.SHORT: 5.0})
        assert result.requirement_micros > 0
        assert only(result.pairings).naked

    def test_robinhood_refuses_a_mismatched_short_call(self):
        result = margin(E.MarginModel.ROBINHOOD, self._legs(100, 400), self.HOLDINGS,
                        marks={self.SHORT: 5.0})
        assert result.disallowed


class TestEquityMargin:
    """
    Stock carried no requirement at all, long or short.

    Shares arrive on every assignment, so this left the entire covered-call,
    PMCC and collar family with an unmargined book while reporting a $0
    requirement and no breach.
    """

    SPOT = 500.0
    SHARES = 1000
    NOTIONAL = SPOT * SHARES

    def test_long_stock_is_charged_reg_t_initial(self):
        result = margin(E.MarginModel.REG_T, [], [], spot=self.SPOT,
                        shares={SYMBOL: self.SHARES})
        assert result.requirement_micros == micros(0.50 * self.NOTIONAL)

    def test_short_stock_is_charged_proceeds_plus_margin(self):
        """A short sale requires 100% of the proceeds plus 50%, so 150%."""
        result = margin(E.MarginModel.REG_T, [], [], spot=self.SPOT,
                        shares={SYMBOL: -self.SHARES})
        assert result.requirement_micros == micros(1.50 * self.NOTIONAL)

    def test_a_cash_account_pays_for_stock_in_full(self):
        result = margin(E.MarginModel.CASH_ACCOUNT, [], [],
                        spot=self.SPOT, shares={SYMBOL: self.SHARES})
        assert result.requirement_micros == micros(self.NOTIONAL)

    def test_a_cash_account_cannot_short_stock(self):
        result = margin(E.MarginModel.CASH_ACCOUNT, [], [],
                        spot=self.SPOT, shares={SYMBOL: -self.SHARES})
        assert result.disallowed
        assert "short" in result.disallowed_reason

    def test_a_covered_call_charges_the_stock_and_nothing_for_the_call(self):
        """
        Reg-T requires margin on the stock and none on the call. Charging the
        option instead of the stock, or neither, are both wrong.
        """
        short = make_contract(1, strike=105.0, expiry_day=30)
        result = margin(E.MarginModel.REG_T, [short], [(1, -1)],
                        spot=100.0, marks={1: 5.0},
                        shares={SYMBOL: 100})
        assert result.requirement_micros == micros(0.50 * 100.0 * 100)
        assert only(result.pairings).covered_by_equity

    def test_flat_equity_is_not_charged(self):
        result = margin(E.MarginModel.REG_T, [], [], spot=self.SPOT,
                        shares={SYMBOL: 0})
        assert result.requirement_micros == 0

    def test_the_charge_scales_with_the_holding(self):
        one = margin(E.MarginModel.REG_T, [], [], spot=self.SPOT,
                     shares={SYMBOL: 100}).requirement_micros
        ten = margin(E.MarginModel.REG_T, [], [], spot=self.SPOT,
                     shares={SYMBOL: 1000}).requirement_micros
        assert ten == 10 * one


class TestPermutationInvariance:
    """
    The same portfolio must margin the same however it was assembled.

    Pairing preferred whichever eligible long it encountered first, which is
    position-id order, which is the order the legs were opened. A strategy that
    legged into a calendar in the "wrong" sequence was refused outright.
    """

    def _legs(self):
        return [
            make_contract(1, strike=100.0, expiry_day=30),    # short, near
            make_contract(2, strike=100.0, expiry_day=400),   # short, far
            make_contract(3, strike=100.0, expiry_day=30),    # long, near
            make_contract(4, strike=100.0, expiry_day=400),   # long, far
        ]

    HOLDINGS = [(3, 1), (4, 1), (1, -1), (2, -1)]
    MARKS = {1: 5.0, 2: 12.0}

    def test_every_permutation_gives_the_same_requirement(self):
        legs = self._legs()
        outcomes = {
            (margin(E.MarginModel.ROBINHOOD, legs, list(perm),
                    marks=self.MARKS).requirement_micros,
             margin(E.MarginModel.ROBINHOOD, legs, list(perm),
                    marks=self.MARKS).disallowed)
            for perm in itertools.permutations(self.HOLDINGS)
        }
        assert len(outcomes) == 1

    def test_a_valid_pairing_is_found_rather_than_refused(self):
        """Near short pairs with near long, far short with far long."""
        result = margin(E.MarginModel.ROBINHOOD, self._legs(), self.HOLDINGS,
                        marks=self.MARKS)
        assert not result.disallowed
        assert result.requirement_micros == 0

    def test_ties_conserve_the_long_dated_long(self):
        """
        With two equal-residual longs available, the shortest-dated short takes
        the earliest-expiring one so a long-dated long stays free for the short
        that actually needs it.
        """
        legs = self._legs()
        result = margin(E.MarginModel.ROBINHOOD, legs, self.HOLDINGS, marks=self.MARKS)
        by_short = {p.short_leg: p.long_leg for p in result.pairings}
        assert by_short[1] == 3
        assert by_short[2] == 4


class TestDisallowedIsVisible:
    def test_a_refused_book_still_reports_that_it_was_refused(self):
        """
        A Disallow verdict contributes nothing to the requirement, so a check of
        the form 'requirement > equity' could never see an impossible book.
        """
        naked = make_contract(1, strike=100.0, expiry_day=30)
        result = margin(E.MarginModel.ROBINHOOD, [naked], [(1, -10)], marks={1: 5.0})
        assert result.disallowed
        assert result.disallowed_reason
