"""
Exact-ledger tests for the structures a strategy actually trades.

Every structure is opened through the engine rather than priced analytically, so
each number asserted is the ledger's own integer money, and each is held to
settlement wherever expiration is part of what defines the structure.
"""
from __future__ import annotations

import pytest

import obt_engine as E
from optionsbacktester.strategy import buy, group, sell
from tests.conftest import EngineHarness, base_config, day_ns, make_bar, make_contract

UNDERLYING = "TEST"
SPOT = 100.0
INITIAL_CASH = 100_000.0
SHARES_PER_CONTRACT = 100

NEAR, FAR, LEAP = 30, 60, 400


def dollars(amount: float) -> int:
    """Microdollars, the engine's exact integer money unit."""
    return round(amount * 1_000_000)


def call_at(cv: int, strike: float, expiry: int = NEAR) -> E.OptionContractVersion:
    return make_contract(cv, strike=strike, expiry_day=expiry)


def put_at(cv: int, strike: float, expiry: int = NEAR) -> E.OptionContractVersion:
    return make_contract(cv, strike=strike, expiry_day=expiry, is_call=False)


def bars(day: int, prices: dict[int, float]) -> list[E.MarketBar]:
    return [make_bar(cv, timestamp_ns=day_ns(day), price=p) for cv, p in prices.items()]


def trade(h: EngineHarness, legs, prices: dict[int, float], *, day: int, spot: float = SPOT):
    """Submit `legs` as one atomic group on `day`; they fill at the next bar's open."""
    h.bar(day_ns(day), bars(day, prices), underlying={UNDERLYING: spot},
          groups=[group(*legs)])
    return h.bar(day_ns(day + 1), bars(day + 1, prices), underlying={UNDERLYING: spot})


def expire(h: EngineHarness, *, day: int, spot: float):
    return h.bar(day_ns(day), [], underlying={UNDERLYING: spot})


def requirement(h: EngineHarness) -> float:
    return h.engine.account_state().margin_requirement


def reasons(h: EngineHarness) -> list[E.RejectReason]:
    return [r.reason for r in h.rejections()]


def reg_t_naked(strike: float, *, is_call: bool, premium: float, spot: float = SPOT) -> float:
    """The 20%/10% CBOE minimum for one uncovered short option, per the rule text."""
    out_of_the_money = max(0.0, strike - spot) if is_call else max(0.0, spot - strike)
    floor = 0.10 * spot if is_call else 0.10 * strike
    return (max(0.20 * spot - out_of_the_money, floor) + premium) * SHARES_PER_CONTRACT


def pairings_of(h: EngineHarness, model: E.MarginModel, spot: float) -> list[E.SpreadPairing]:
    """How the margin model pairs whatever the engine currently holds."""
    return E.evaluate_margin(
        model,
        list(h.contracts.values()),
        [(p.contract_version_id, p.quantity) for p in h.positions()],
        {UNDERLYING: spot},
        {},
        {e.symbol: e.shares for e in h.equity_positions()},
    ).pairings


@pytest.fixture
def account():
    """
    Factory for harnesses whose ledgers are all reconciled at teardown.

    Reconciliation is an invariant of every structure rather than a property of
    any one of them, so it is asserted here instead of in each test.
    """
    built: list[EngineHarness] = []

    def build(contracts, **cfg) -> EngineHarness:
        h = EngineHarness(
            base_config(spread=E.SpreadModelKind.ZERO, fees=False, **cfg), contracts)
        built.append(h)
        return h

    yield build

    for h in built:
        assert h.engine.ledger_reconciles()
        assert h.finalize().ledger_reconciles


class TestLongOptions:
    @pytest.mark.parametrize("is_call, premium", [(True, 5.00), (False, 4.00)],
                             ids=["long_call", "long_put"])
    def test_long_option_pays_the_debit_and_requires_no_margin(self, account, is_call, premium):
        h = account([call_at(1, 100.0) if is_call else put_at(1, 100.0)])
        trade(h, [buy(1)], {1: premium}, day=1)

        (position,) = h.positions()
        assert position.quantity == 1
        assert position.cost_basis_micros == dollars(premium * SHARES_PER_CONTRACT)
        assert h.cash_micros == dollars(INITIAL_CASH - premium * SHARES_PER_CONTRACT)
        assert requirement(h) == 0.0


class TestNakedShorts:
    @pytest.mark.parametrize("is_call, premium", [(True, 3.00), (False, 4.00)],
                             ids=["short_call", "short_put"])
    def test_short_option_credits_the_premium_and_charges_the_reg_t_minimum(
        self, account, is_call, premium
    ):
        h = account([call_at(1, 100.0) if is_call else put_at(1, 100.0)],
                    margin=E.MarginModel.REG_T)
        trade(h, [sell(1)], {1: premium}, day=1)

        (position,) = h.positions()
        assert position.quantity == -1
        assert position.cost_basis_micros == -dollars(premium * SHARES_PER_CONTRACT)
        assert h.cash_micros == dollars(INITIAL_CASH + premium * SHARES_PER_CONTRACT)
        assert requirement(h) == pytest.approx(
            reg_t_naked(100.0, is_call=is_call, premium=premium))

    def test_uncovered_short_call_needs_reg_t_because_robinhood_refuses_it(self, account):
        h = account([call_at(1, 100.0)], margin=E.MarginModel.ROBINHOOD)
        trade(h, [sell(1)], {1: 3.00}, day=1)

        assert reasons(h) == [E.RejectReason.BROKER_DISALLOWED]
        assert h.cash_micros == dollars(INITIAL_CASH)


class TestCallVerticals:
    LOWER, UPPER = 1, 2
    LOWER_STRIKE, UPPER_STRIKE = 95.0, 105.0
    WIDTH = UPPER_STRIKE - LOWER_STRIKE
    PRICES = {LOWER: 7.00, UPPER: 2.00}

    def universe(self):
        return [call_at(self.LOWER, self.LOWER_STRIKE), call_at(self.UPPER, self.UPPER_STRIKE)]

    def test_debit_vertical_pays_the_net_debit_and_requires_nothing_beyond_it(self, account):
        h = account(self.universe())
        trade(h, [buy(self.LOWER), sell(self.UPPER)], self.PRICES, day=1)

        assert h.quantity_of(self.LOWER) == 1
        assert h.quantity_of(self.UPPER) == -1
        assert h.cash_micros == dollars(INITIAL_CASH - 5.00 * SHARES_PER_CONTRACT)
        assert requirement(h) == 0.0

    def test_debit_vertical_is_worth_exactly_the_strike_width_above_both_strikes(self, account):
        h = account(self.universe())
        trade(h, [buy(self.LOWER), sell(self.UPPER)], self.PRICES, day=1)
        expire(h, day=NEAR, spot=120.0)

        assert h.shares_of(UNDERLYING) == 0
        assert h.cash_micros == dollars(
            INITIAL_CASH - 5.00 * SHARES_PER_CONTRACT + self.WIDTH * SHARES_PER_CONTRACT)
        assert h.positions() == []

    def test_credit_vertical_collects_the_credit_and_is_charged_the_strike_width(self, account):
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, [sell(self.LOWER), buy(self.UPPER)], self.PRICES, day=1)

        assert h.cash_micros == dollars(INITIAL_CASH + 5.00 * SHARES_PER_CONTRACT)
        assert requirement(h) == pytest.approx(self.WIDTH * SHARES_PER_CONTRACT)

    def test_credit_vertical_loses_exactly_the_width_above_both_strikes(self, account):
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, [sell(self.LOWER), buy(self.UPPER)], self.PRICES, day=1)
        expire(h, day=NEAR, spot=120.0)

        assert h.shares_of(UNDERLYING) == 0
        assert h.cash_micros == dollars(
            INITIAL_CASH + 5.00 * SHARES_PER_CONTRACT - self.WIDTH * SHARES_PER_CONTRACT)


class TestPutVerticals:
    UPPER, LOWER = 1, 2
    UPPER_STRIKE, LOWER_STRIKE = 105.0, 95.0
    WIDTH = UPPER_STRIKE - LOWER_STRIKE
    PRICES = {UPPER: 7.00, LOWER: 2.00}

    def universe(self):
        return [put_at(self.UPPER, self.UPPER_STRIKE), put_at(self.LOWER, self.LOWER_STRIKE)]

    def test_debit_vertical_pays_the_net_debit_and_requires_nothing_beyond_it(self, account):
        h = account(self.universe())
        trade(h, [buy(self.UPPER), sell(self.LOWER)], self.PRICES, day=1)

        assert h.quantity_of(self.UPPER) == 1
        assert h.quantity_of(self.LOWER) == -1
        assert h.cash_micros == dollars(INITIAL_CASH - 5.00 * SHARES_PER_CONTRACT)
        assert requirement(h) == 0.0

    def test_debit_vertical_is_worth_exactly_the_strike_width_below_both_strikes(self, account):
        h = account(self.universe())
        trade(h, [buy(self.UPPER), sell(self.LOWER)], self.PRICES, day=1)
        expire(h, day=NEAR, spot=80.0)

        assert h.shares_of(UNDERLYING) == 0
        assert h.cash_micros == dollars(
            INITIAL_CASH - 5.00 * SHARES_PER_CONTRACT + self.WIDTH * SHARES_PER_CONTRACT)

    def test_credit_vertical_collects_the_credit_and_is_charged_the_strike_width(self, account):
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, [sell(self.UPPER), buy(self.LOWER)], self.PRICES, day=1)

        assert h.cash_micros == dollars(INITIAL_CASH + 5.00 * SHARES_PER_CONTRACT)
        assert requirement(h) == pytest.approx(self.WIDTH * SHARES_PER_CONTRACT)


class TestCalendarSpread:
    LONG, SHORT = 1, 2
    PRICES = {LONG: 8.00, SHORT: 4.00}
    DEBIT = 4.00

    def universe(self):
        return [call_at(self.LONG, 100.0, FAR), call_at(self.SHORT, 100.0, NEAR)]

    def test_calendar_pays_the_net_debit_and_requires_nothing_beyond_it(self, account):
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, [buy(self.LONG), sell(self.SHORT)], self.PRICES, day=1)

        assert h.quantity_of(self.LONG) == 1
        assert h.quantity_of(self.SHORT) == -1
        assert h.cash_micros == dollars(INITIAL_CASH - self.DEBIT * SHARES_PER_CONTRACT)
        assert requirement(h) == 0.0

    def test_near_leg_expiring_at_the_money_leaves_the_long_leg_untouched(self, account):
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, [buy(self.LONG), sell(self.SHORT)], self.PRICES, day=1)
        expire(h, day=NEAR, spot=SPOT)

        assert h.quantity_of(self.SHORT) == 0
        (position,) = h.positions()
        assert position.contract_version_id == self.LONG
        assert position.cost_basis_micros == dollars(8.00 * SHARES_PER_CONTRACT)
        assert h.cash_micros == dollars(INITIAL_CASH - self.DEBIT * SHARES_PER_CONTRACT)


class TestDiagonalSpreads:
    """
    A long call that is lower struck and longer lived than the short caps the
    loss at the debit, so the short is covered by the long rather than naked --
    which is what makes a poor man's covered call a retail-permitted position.
    """

    LONG, SHORT = 1, 2
    PRICES = {LONG: 14.00, SHORT: 2.00}
    DEBIT = 12.00

    def universe(self):
        return [call_at(self.LONG, 90.0, FAR), call_at(self.SHORT, 105.0, NEAR)]

    def test_pmcc_pays_the_debit_and_requires_nothing_beyond_it(self, account):
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, [buy(self.LONG), sell(self.SHORT)], self.PRICES, day=1)

        assert len(h.fills()) == 2
        assert h.cash_micros == dollars(INITIAL_CASH - self.DEBIT * SHARES_PER_CONTRACT)
        assert requirement(h) == 0.0

    def test_pmcc_is_permitted_at_a_retail_broker_while_holding_no_shares(self, account):
        h = account(self.universe(), margin=E.MarginModel.ROBINHOOD)
        trade(h, [buy(self.LONG), sell(self.SHORT)], self.PRICES, day=1)

        assert h.shares_of(UNDERLYING) == 0
        assert h.quantity_of(self.SHORT) == -1
        assert not h.rejections()
        assert requirement(h) == 0.0

    def test_the_same_short_call_alone_is_refused_at_that_retail_broker(self, account):
        h = account(self.universe(), margin=E.MarginModel.ROBINHOOD)
        trade(h, [sell(self.SHORT)], {self.SHORT: 2.00}, day=1)

        assert reasons(h) == [E.RejectReason.BROKER_DISALLOWED]
        assert h.cash_micros == dollars(INITIAL_CASH)


class TestItmDiagonal:
    """
    Short below the long leaves residual risk: above the long strike the short
    is in the money by the strike difference and the long is not yet worth
    anything, so the requirement is that difference.
    """

    LONG, SHORT = 1, 2
    LONG_STRIKE, SHORT_STRIKE = 105.0, 95.0
    PRICES = {LONG: 9.00, SHORT: 7.00}

    def universe(self):
        return [call_at(self.LONG, self.LONG_STRIKE, FAR),
                call_at(self.SHORT, self.SHORT_STRIKE, NEAR)]

    @pytest.mark.parametrize("model", [E.MarginModel.REG_T, E.MarginModel.ROBINHOOD])
    def test_requirement_is_the_strike_difference(self, account, model):
        h = account(self.universe(), margin=model)
        trade(h, [buy(self.LONG), sell(self.SHORT)], self.PRICES, day=1)

        assert len(h.fills()) == 2
        assert h.cash_micros == dollars(INITIAL_CASH - 2.00 * SHARES_PER_CONTRACT)
        assert requirement(h) == pytest.approx(
            (self.LONG_STRIKE - self.SHORT_STRIKE) * SHARES_PER_CONTRACT)


class TestPoorMansCoveredCallRewriting:
    """
    One long LEAPS backs a succession of short calls.

    Each short is written only after the previous one expires, so the LEAPS is
    never paired with more than one short at a time and every sale is permitted
    at a retail broker despite the account holding no shares.
    """

    LEAPS = 1
    SHORTS = ((2, NEAR), (3, FAR), (4, 90))
    LEAPS_DEBIT, CREDIT = 14.00, 2.00

    def universe(self):
        return [call_at(self.LEAPS, 90.0, LEAP)] + [
            call_at(cv, 105.0, expiry) for cv, expiry in self.SHORTS]

    def written(self, account) -> EngineHarness:
        """Writes all three shorts, letting each of the first two expire worthless."""
        (first, _), (second, second_expiry), (third, _) = self.SHORTS
        h = account(self.universe(), margin=E.MarginModel.ROBINHOOD)

        trade(h, [buy(self.LEAPS), sell(first)],
              {self.LEAPS: self.LEAPS_DEBIT, first: self.CREDIT}, day=1)
        assert h.quantity_of(first) == -1

        expire(h, day=NEAR, spot=SPOT)
        assert h.quantity_of(first) == 0
        assert h.quantity_of(self.LEAPS) == 1

        trade(h, [sell(second)], {second: self.CREDIT}, day=NEAR + 1)
        assert h.quantity_of(second) == -1

        expire(h, day=second_expiry, spot=SPOT)
        assert h.quantity_of(second) == 0
        assert h.quantity_of(self.LEAPS) == 1

        trade(h, [sell(third)], {third: self.CREDIT}, day=second_expiry + 1)
        return h

    def test_three_shorts_are_written_in_turn_against_one_long_leaps(self, account):
        h = self.written(account)
        third = self.SHORTS[-1][0]

        assert not h.rejections()
        assert h.quantity_of(self.LEAPS) == 1
        assert h.quantity_of(third) == -1
        assert len([f for f in h.fills() if f.side == E.OrderSide.SELL]) == 3

    def test_total_credit_collected_is_the_exact_sum_of_the_three_sales(self, account):
        h = self.written(account)

        sales = [f for f in h.fills() if f.side == E.OrderSide.SELL]
        assert sum(f.net_cash_micros for f in sales) == dollars(
            3 * self.CREDIT * SHARES_PER_CONTRACT)
        assert h.cash_micros == dollars(
            INITIAL_CASH - self.LEAPS_DEBIT * SHARES_PER_CONTRACT
            + 3 * self.CREDIT * SHARES_PER_CONTRACT)

    def test_the_long_leg_keeps_its_original_basis_across_every_rewrite(self, account):
        h = self.written(account)

        leaps = next(p for p in h.positions() if p.contract_version_id == self.LEAPS)
        assert leaps.cost_basis_micros == dollars(self.LEAPS_DEBIT * SHARES_PER_CONTRACT)
        assert leaps.realized_pnl == 0.0


class TestStraddleAndStrangle:
    CALL, PUT = 1, 2

    @pytest.mark.parametrize("call_strike, put_strike, call_price, put_price", [
        pytest.param(100.0, 100.0, 5.00, 4.00, id="straddle"),
        pytest.param(105.0, 95.0, 2.00, 3.00, id="strangle"),
    ])
    def test_long_combination_pays_the_sum_of_both_debits(
        self, account, call_strike, put_strike, call_price, put_price
    ):
        h = account([call_at(self.CALL, call_strike), put_at(self.PUT, put_strike)])
        trade(h, [buy(self.CALL), buy(self.PUT)],
              {self.CALL: call_price, self.PUT: put_price}, day=1)

        assert h.quantity_of(self.CALL) == h.quantity_of(self.PUT) == 1
        assert h.cash_micros == dollars(
            INITIAL_CASH - (call_price + put_price) * SHARES_PER_CONTRACT)
        assert requirement(h) == 0.0

    def test_short_strangle_is_charged_the_naked_minimum_on_each_side(self, account):
        h = account([call_at(self.CALL, 105.0), put_at(self.PUT, 95.0)],
                    margin=E.MarginModel.REG_T)
        trade(h, [sell(self.CALL), sell(self.PUT)], {self.CALL: 2.00, self.PUT: 3.00}, day=1)

        assert h.cash_micros == dollars(INITIAL_CASH + 5.00 * SHARES_PER_CONTRACT)
        assert requirement(h) == pytest.approx(
            reg_t_naked(105.0, is_call=True, premium=2.00)
            + reg_t_naked(95.0, is_call=False, premium=3.00))


class TestButterfly:
    LOW, MID, HIGH = 1, 2, 3
    PRICES = {LOW: 7.00, MID: 4.00, HIGH: 2.00}
    DEBIT = 1.00

    def universe(self):
        return [call_at(self.LOW, 95.0), call_at(self.MID, 100.0), call_at(self.HIGH, 105.0)]

    def legs(self):
        return [buy(self.LOW), sell(self.MID, 2), buy(self.HIGH)]

    def test_butterfly_pays_the_net_debit_for_the_four_contracts(self, account):
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, self.legs(), self.PRICES, day=1)

        assert [h.quantity_of(cv) for cv in (self.LOW, self.MID, self.HIGH)] == [1, -2, 1]
        assert h.cash_micros == dollars(INITIAL_CASH - self.DEBIT * SHARES_PER_CONTRACT)

    def test_butterfly_requires_no_margin_at_all(self, account):
        """
        Max-loss netting produces zero, not the wing width.

        The two short middles pair separately -- one against the lower long at
        zero residual, one against the upper long at the wing width -- but the
        charge comes from netting all four legs jointly, and a long butterfly's
        payoff is non-negative at every strike in it. Its only loss is the debit
        already paid in cash, which is not something margin has to secure.
        """
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, self.legs(), self.PRICES, day=1)

        assert requirement(h) == 0.0

    def test_butterfly_expiring_below_every_strike_loses_only_the_debit(self, account):
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, self.legs(), self.PRICES, day=1)
        expire(h, day=NEAR, spot=90.0)

        assert h.positions() == []
        assert h.shares_of(UNDERLYING) == 0
        assert h.cash_micros == dollars(INITIAL_CASH - self.DEBIT * SHARES_PER_CONTRACT)
        assert h.finalize().net_pnl == -self.DEBIT * SHARES_PER_CONTRACT


class TestIronCondor:
    LONG_PUT, SHORT_PUT, SHORT_CALL, LONG_CALL = 1, 2, 3, 4
    PUT_WIDTH, CALL_WIDTH = 5.0, 10.0
    OPEN_PRICES = {LONG_PUT: 1.00, SHORT_PUT: 3.00, SHORT_CALL: 3.00, LONG_CALL: 0.50}
    CLOSE_PRICES = {LONG_PUT: 0.50, SHORT_PUT: 1.00, SHORT_CALL: 1.00, LONG_CALL: 0.25}
    CREDIT = 4.50
    CLOSING_DEBIT = 1.25

    def universe(self):
        return [put_at(self.LONG_PUT, 90.0), put_at(self.SHORT_PUT, 95.0),
                call_at(self.SHORT_CALL, 105.0), call_at(self.LONG_CALL, 115.0)]

    def legs(self):
        return [buy(self.LONG_PUT), sell(self.SHORT_PUT),
                sell(self.SHORT_CALL), buy(self.LONG_CALL)]

    def opened(self, account) -> EngineHarness:
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, self.legs(), self.OPEN_PRICES, day=1)
        return h

    def test_condor_collects_the_net_credit_of_all_four_legs(self, account):
        h = self.opened(account)

        assert [h.quantity_of(cv) for cv in
                (self.LONG_PUT, self.SHORT_PUT, self.SHORT_CALL, self.LONG_CALL)] == [1, -1, -1, 1]
        assert h.cash_micros == dollars(INITIAL_CASH + self.CREDIT * SHARES_PER_CONTRACT)

    def test_condor_is_charged_the_wider_wing_only_and_not_the_sum(self, account):
        """Only one wing can finish in the money, so charging both would secure
        a loss that cannot occur."""
        h = self.opened(account)

        assert requirement(h) == pytest.approx(
            max(self.PUT_WIDTH, self.CALL_WIDTH) * SHARES_PER_CONTRACT)
        assert requirement(h) < (self.PUT_WIDTH + self.CALL_WIDTH) * SHARES_PER_CONTRACT

    def test_closing_all_four_legs_in_one_group_nets_credit_minus_debit(self, account):
        h = self.opened(account)
        closing = [sell(self.LONG_PUT, reduce_only=True), buy(self.SHORT_PUT, reduce_only=True),
                   buy(self.SHORT_CALL, reduce_only=True), sell(self.LONG_CALL, reduce_only=True)]
        trade(h, closing, self.CLOSE_PRICES, day=5)

        assert len(h.fills()) == 8
        assert h.positions() == []
        assert h.cash_micros == dollars(
            INITIAL_CASH + (self.CREDIT - self.CLOSING_DEBIT) * SHARES_PER_CONTRACT)
        assert requirement(h) == 0.0


class TestIronButterfly:
    LONG_PUT, SHORT_PUT, SHORT_CALL, LONG_CALL = 1, 2, 3, 4
    WING_WIDTH = 10.0
    PRICES = {LONG_PUT: 1.00, SHORT_PUT: 5.00, SHORT_CALL: 5.00, LONG_CALL: 1.00}
    CREDIT = 8.00

    def universe(self):
        return [put_at(self.LONG_PUT, 90.0), put_at(self.SHORT_PUT, 100.0),
                call_at(self.SHORT_CALL, 100.0), call_at(self.LONG_CALL, 110.0)]

    def opened(self, account) -> EngineHarness:
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, [buy(self.LONG_PUT), sell(self.SHORT_PUT),
                  sell(self.SHORT_CALL), buy(self.LONG_CALL)], self.PRICES, day=1)
        return h

    def test_iron_butterfly_collects_the_short_straddle_credit_net_of_the_wings(self, account):
        h = self.opened(account)

        assert h.quantity_of(self.SHORT_PUT) == h.quantity_of(self.SHORT_CALL) == -1
        assert h.quantity_of(self.LONG_PUT) == h.quantity_of(self.LONG_CALL) == 1
        assert h.cash_micros == dollars(INITIAL_CASH + self.CREDIT * SHARES_PER_CONTRACT)

    def test_iron_butterfly_is_charged_one_wing_width(self, account):
        h = self.opened(account)

        assert requirement(h) == pytest.approx(self.WING_WIDTH * SHARES_PER_CONTRACT)

    def test_iron_butterfly_pinned_at_the_body_keeps_the_whole_credit(self, account):
        h = self.opened(account)
        expire(h, day=NEAR, spot=SPOT)

        assert h.positions() == []
        assert h.shares_of(UNDERLYING) == 0
        assert h.cash_micros == dollars(INITIAL_CASH + self.CREDIT * SHARES_PER_CONTRACT)


class TestRatioSpread:
    LONG, SHORT = 1, 2
    SHORT_STRIKE = 110.0
    PRICES = {LONG: 5.00, SHORT: 1.50}

    def universe(self):
        return [call_at(self.LONG, 100.0), call_at(self.SHORT, self.SHORT_STRIKE)]

    def legs(self):
        return [buy(self.LONG), sell(self.SHORT, 2)]

    def test_retail_broker_refuses_the_ratio_because_the_second_short_is_uncovered(
        self, account
    ):
        h = account(self.universe(), margin=E.MarginModel.ROBINHOOD)
        trade(h, self.legs(), self.PRICES, day=1)

        assert reasons(h) == [E.RejectReason.BROKER_DISALLOWED] * 2
        assert h.positions() == []
        assert h.cash_micros == dollars(INITIAL_CASH)

    def test_reg_t_charges_the_naked_minimum_on_the_one_unpaired_contract(self, account):
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, self.legs(), self.PRICES, day=1)

        naked = reg_t_naked(self.SHORT_STRIKE, is_call=True, premium=1.50)
        assert h.quantity_of(self.SHORT) == -2
        assert h.cash_micros == dollars(INITIAL_CASH - 2.00 * SHARES_PER_CONTRACT)
        assert requirement(h) == pytest.approx(naked)
        assert requirement(h) < 2 * naked


class TestEquityBackedStructures:
    """
    Structures whose stock leg is acquired by exercising a long call at
    expiration, which is how a backtest ends up holding shares at all.
    """

    LONG_CALL, WRITTEN_CALL, PROTECTIVE_PUT = 1, 2, 3
    CALL_PREMIUM, STRIKE, SPOT_AT_EXPIRY = 5.00, 100.0, 110.0

    def universe(self):
        return [call_at(self.LONG_CALL, self.STRIKE, NEAR),
                call_at(self.WRITTEN_CALL, 120.0, FAR),
                put_at(self.PROTECTIVE_PUT, 105.0, FAR)]

    def with_shares(self, account, model=E.MarginModel.ROBINHOOD) -> EngineHarness:
        """Exercises a long ITM call, leaving 100 shares and the strike paid out."""
        h = account(self.universe(), margin=model)
        trade(h, [buy(self.LONG_CALL)], {self.LONG_CALL: self.CALL_PREMIUM}, day=1)
        expire(h, day=NEAR, spot=self.SPOT_AT_EXPIRY)
        assert h.shares_of(UNDERLYING) == SHARES_PER_CONTRACT
        return h

    def test_call_written_against_exercised_shares_charges_only_the_stock(self, account):
        h = self.with_shares(account)
        trade(h, [sell(self.WRITTEN_CALL)], {self.WRITTEN_CALL: 2.00},
              day=NEAR + 1, spot=self.SPOT_AT_EXPIRY)

        assert h.quantity_of(self.WRITTEN_CALL) == -1
        assert h.shares_of(UNDERLYING) == SHARES_PER_CONTRACT
        # Reg-T margins the stock at 50% and charges nothing for the call, so the
        # covered call's requirement is the stock's, not zero.
        assert requirement(h) == pytest.approx(
            0.50 * self.SPOT_AT_EXPIRY * SHARES_PER_CONTRACT)
        assert h.cash_micros == dollars(
            INITIAL_CASH - self.CALL_PREMIUM * SHARES_PER_CONTRACT
            - self.STRIKE * SHARES_PER_CONTRACT + 2.00 * SHARES_PER_CONTRACT)

        (pairing,) = pairings_of(h, E.MarginModel.ROBINHOOD, self.SPOT_AT_EXPIRY)
        assert pairing.covered_by_equity
        assert pairing.requirement == 0.0

    def test_protective_put_over_exercised_shares_charges_only_the_stock(self, account):
        h = self.with_shares(account)
        trade(h, [buy(self.PROTECTIVE_PUT)], {self.PROTECTIVE_PUT: 3.00},
              day=NEAR + 1, spot=self.SPOT_AT_EXPIRY)

        assert h.quantity_of(self.PROTECTIVE_PUT) == 1
        assert h.shares_of(UNDERLYING) == SHARES_PER_CONTRACT
        # The long put costs its premium and adds no requirement; the stock is
        # margined in its own right.
        assert requirement(h) == pytest.approx(
            0.50 * self.SPOT_AT_EXPIRY * SHARES_PER_CONTRACT)
        assert h.cash_micros == dollars(
            INITIAL_CASH - self.CALL_PREMIUM * SHARES_PER_CONTRACT
            - self.STRIKE * SHARES_PER_CONTRACT - 3.00 * SHARES_PER_CONTRACT)

    def test_protective_put_exercised_sells_the_shares_at_its_strike(self, account):
        """The put's strike is the floor: the shares leave at 105 no matter how
        far below it the underlying settles."""
        h = self.with_shares(account)
        trade(h, [buy(self.PROTECTIVE_PUT)], {self.PROTECTIVE_PUT: 3.00},
              day=NEAR + 1, spot=self.SPOT_AT_EXPIRY)
        expire(h, day=FAR, spot=80.0)

        assert h.shares_of(UNDERLYING) == 0
        assert h.equity_positions() == []
        assert h.positions() == []
        assert h.cash_micros == dollars(
            INITIAL_CASH - self.CALL_PREMIUM * SHARES_PER_CONTRACT
            - self.STRIKE * SHARES_PER_CONTRACT - 3.00 * SHARES_PER_CONTRACT
            + 105.0 * SHARES_PER_CONTRACT)


class TestRoll:
    NEARER, FURTHER = 1, 2

    def universe(self):
        return [call_at(self.NEARER, 105.0, NEAR), call_at(self.FURTHER, 105.0, FAR)]

    def test_rolling_a_short_call_out_in_time_fills_both_legs_together(self, account):
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, [sell(self.NEARER)], {self.NEARER: 3.00}, day=1)

        trade(h, [buy(self.NEARER, reduce_only=True), sell(self.FURTHER)],
              {self.NEARER: 1.00, self.FURTHER: 2.50}, day=10)

        roll = h.fills()[-2:]
        assert len({f.group_id for f in roll}) == 1
        assert len({f.filled_at for f in roll}) == 1
        assert h.quantity_of(self.NEARER) == 0
        assert h.quantity_of(self.FURTHER) == -1

    def test_roll_nets_exactly_the_difference_between_the_two_premiums(self, account):
        h = account(self.universe(), margin=E.MarginModel.REG_T)
        trade(h, [sell(self.NEARER)], {self.NEARER: 3.00}, day=1)
        trade(h, [buy(self.NEARER, reduce_only=True), sell(self.FURTHER)],
              {self.NEARER: 1.00, self.FURTHER: 2.50}, day=10)

        assert sum(f.net_cash_micros for f in h.fills()[-2:]) == dollars(
            1.50 * SHARES_PER_CONTRACT)
        assert h.cash_micros == dollars(INITIAL_CASH + 4.50 * SHARES_PER_CONTRACT)


class TestGroupRefusalIsAtomic:
    LONG_PUT, SHORT_PUT, SHORT_CALL, EXTRA_PUT = 1, 2, 3, 4
    PRICES = {LONG_PUT: 1.00, SHORT_PUT: 3.00, SHORT_CALL: 3.00, EXTRA_PUT: 0.50}

    def test_four_leg_group_is_refused_whole_when_one_leg_leaves_a_naked_call(self, account):
        """
        The intended condor's upper wing is a put instead of a call, so the short
        call has nothing above it and the group is refused rather than partially
        executed into an uncovered short.
        """
        h = account([put_at(self.LONG_PUT, 90.0), put_at(self.SHORT_PUT, 95.0),
                     call_at(self.SHORT_CALL, 105.0), put_at(self.EXTRA_PUT, 85.0)],
                    margin=E.MarginModel.ROBINHOOD)
        trade(h, [buy(self.LONG_PUT), sell(self.SHORT_PUT),
                  sell(self.SHORT_CALL), buy(self.EXTRA_PUT)], self.PRICES, day=1)

        assert len(h.fills()) == 0
        assert reasons(h) == [E.RejectReason.BROKER_DISALLOWED] * 4
        assert h.positions() == []
        assert h.engine.account_state().open_position_count == 0
        assert h.cash_micros == dollars(INITIAL_CASH)
