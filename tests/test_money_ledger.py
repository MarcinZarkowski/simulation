"""
Money representation and ledger integrity.

The previous engine held cash in float32, where one ULP is $1.00 at a $10M
balance, so long runs silently lost money that no test could detect. These tests
assert exactness rather than approximate equality: with integer microdollars,
"the ledger balances" is an equality, not a tolerance.
"""
from __future__ import annotations

import pytest

import obt_engine as E
from optionsbacktester.strategy import buy, group, sell
from tests.conftest import (
    EngineHarness,
    base_config,
    day_ns,
    make_bar,
    make_contract,
)

CALL = 1
PUT = 2
MICROS = 1_000_000


class TestMoneyRepresentation:
    def test_dollars_round_trip_through_microdollars(self):
        for dollars in (0.0, 0.01, 1.0, 5.55, 123.45, 99999.99):
            assert E.Money.from_dollars(dollars).to_dollars() == pytest.approx(dollars, abs=1e-9)

    def test_one_dollar_is_exactly_one_million_micros(self):
        assert E.Money.from_dollars(1.0).micros == 1_000_000

    def test_a_penny_is_exact(self):
        assert E.Money.from_dollars(0.01).micros == 10_000

    def test_conversion_rounds_to_the_nearest_microdollar(self):
        assert E.Money.from_dollars(0.0000004).micros == 0
        assert E.Money.from_dollars(0.0000006).micros == 1

    def test_negative_amounts_are_representable(self):
        assert E.Money.from_dollars(-12.34).micros == -12_340_000

    def test_large_balances_keep_cent_precision(self):
        """
        The case float32 could not represent. At $10M its ULP is $1.00, so a
        one-cent difference would vanish.
        """
        big = E.Money.from_dollars(10_000_000.00)
        bigger = E.Money.from_dollars(10_000_000.01)
        assert bigger.micros - big.micros == 10_000

    def test_hundred_million_keeps_cent_precision(self):
        a = E.Money.from_dollars(100_000_000.00)
        b = E.Money.from_dollars(100_000_000.01)
        assert b.micros - a.micros == 10_000


class TestPremiumArithmetic:
    """Premium times multiplier times contracts must be exact, not rounded."""

    @staticmethod
    def _buy_once(price: float, contracts: int = 1, multiplier: int = 100) -> EngineHarness:
        c = make_contract(CALL, strike=100.0, expiry_day=60, multiplier=multiplier,
                          deliverable_shares=multiplier)
        h = EngineHarness(base_config(cash=1_000_000.0), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=price)],
              groups=[group(buy(CALL, contracts))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=price)])
        return h

    @pytest.mark.parametrize(("price", "contracts"), [
        (1.00, 1), (0.01, 1), (5.55, 1), (12.34, 3), (0.05, 10), (99.99, 2),
    ])
    def test_debit_is_exactly_price_times_multiplier_times_contracts(self, price, contracts):
        h = self._buy_once(price, contracts)
        expected = 1_000_000 * MICROS - round(price * 100 * contracts * MICROS)
        assert h.cash_micros == expected

    def test_cost_basis_equals_the_cash_paid(self):
        h = self._buy_once(5.55, 2)
        position = h.positions()[0]
        assert position.cost_basis_micros == round(5.55 * 100 * 2 * MICROS)

    def test_non_standard_multiplier_is_respected(self):
        """An adjusted contract delivering 10 shares must not be charged for 100."""
        h = self._buy_once(5.00, 1, multiplier=10)
        assert h.cash_micros == 1_000_000 * MICROS - round(5.00 * 10 * MICROS)


class TestRoundTripExactness:
    def test_buy_then_sell_at_the_same_price_returns_the_starting_cash(self):
        c = make_contract(CALL, strike=100.0, expiry_day=60)
        h = EngineHarness(base_config(cash=50_000.0), [c])
        start = h.cash_micros

        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=7.77)],
              groups=[group(buy(CALL, 3))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=7.77)],
              groups=[group(sell(CALL, 3, reduce_only=True))])
        h.bar(day_ns(3), [make_bar(CALL, timestamp_ns=day_ns(3), price=7.77)])

        assert h.cash_micros == start
        assert h.quantity_of(CALL) == 0

    def test_many_round_trips_do_not_drift(self):
        """
        Twenty cycles at an awkward price. Any per-trade rounding error would
        accumulate visibly; with integer money the total is exact.
        """
        c = make_contract(CALL, strike=100.0, expiry_day=400)
        h = EngineHarness(base_config(cash=100_000.0), [c])
        start = h.cash_micros

        day = 1
        for _ in range(20):
            h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=3.33)],
                  groups=[group(buy(CALL, 1))])
            day += 1
            h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=3.33)],
                  groups=[group(sell(CALL, 1, reduce_only=True))])
            day += 1
        h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=3.33)])

        assert h.cash_micros == start

    def test_profitable_round_trip_realizes_the_exact_difference(self):
        """
        Fills land on the bar after the signal, so the entry price is bar 2's
        open and the exit price is bar 3's open.
        """
        c = make_contract(CALL, strike=100.0, expiry_day=60)
        h = EngineHarness(base_config(cash=50_000.0), [c])
        start = h.cash_micros

        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=2.00)],
              groups=[group(buy(CALL, 1))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=2.00)],
              groups=[group(sell(CALL, 1, reduce_only=True))])
        h.bar(day_ns(3), [make_bar(CALL, timestamp_ns=day_ns(3), price=3.50)])

        assert h.cash_micros - start == round(1.50 * 100 * MICROS)


class TestPartialCloses:
    def test_partial_close_leaves_the_remaining_quantity(self):
        c = make_contract(CALL, strike=100.0, expiry_day=60)
        h = EngineHarness(base_config(), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=4.00)],
              groups=[group(buy(CALL, 5))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=4.00)],
              groups=[group(sell(CALL, 2, reduce_only=True))])
        h.bar(day_ns(3), [make_bar(CALL, timestamp_ns=day_ns(3), price=6.00)])
        assert h.quantity_of(CALL) == 3

    def test_partial_close_releases_a_proportional_basis(self):
        """
        Closing 2 of 5 contracts must release exactly two fifths of the basis, so
        the remainder still carries its original average cost.
        """
        c = make_contract(CALL, strike=100.0, expiry_day=60)
        h = EngineHarness(base_config(), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=4.00)],
              groups=[group(buy(CALL, 5))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=4.00)],
              groups=[group(sell(CALL, 2, reduce_only=True))])
        h.bar(day_ns(3), [make_bar(CALL, timestamp_ns=day_ns(3), price=6.00,
                                   open_price=6.00)])

        position = h.positions()[0]
        assert position.cost_basis_micros == round(3 * 4.00 * 100 * MICROS)
        assert position.realized_pnl == pytest.approx(2 * 2.00 * 100)

    def test_averaging_into_a_position_blends_the_basis(self):
        c = make_contract(CALL, strike=100.0, expiry_day=60)
        h = EngineHarness(base_config(), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=2.00)],
              groups=[group(buy(CALL, 1))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=2.00)],
              groups=[group(buy(CALL, 1))])
        h.bar(day_ns(3), [make_bar(CALL, timestamp_ns=day_ns(3), price=4.00,
                                   open_price=4.00)])

        position = h.positions()[0]
        assert position.quantity == 2
        assert position.cost_basis_micros == round(6.00 * 100 * MICROS)
        assert position.average_cost == pytest.approx(3.00 * 100)

    def test_selling_through_zero_opens_a_short(self):
        c = make_contract(CALL, strike=100.0, expiry_day=60)
        h = EngineHarness(base_config(margin=E.MarginModel.REG_T), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=3.00)],
              groups=[group(buy(CALL, 1))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=3.00)],
              groups=[group(sell(CALL, 3))])
        h.bar(day_ns(3), [make_bar(CALL, timestamp_ns=day_ns(3), price=3.00)])
        assert h.quantity_of(CALL) == -2


class TestShortPositions:
    def test_selling_credits_the_premium(self):
        c = make_contract(CALL, strike=110.0, expiry_day=60)
        h = EngineHarness(base_config(cash=100_000.0, margin=E.MarginModel.REG_T), [c])
        start = h.cash_micros
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=1.50)],
              groups=[group(sell(CALL, 2))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=1.50)])
        assert h.cash_micros - start == round(1.50 * 100 * 2 * MICROS)

    def test_short_basis_is_negative(self):
        c = make_contract(CALL, strike=110.0, expiry_day=60)
        h = EngineHarness(base_config(margin=E.MarginModel.REG_T), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=1.50)],
              groups=[group(sell(CALL, 1))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=1.50)])
        assert h.positions()[0].cost_basis_micros == -round(1.50 * 100 * MICROS)

    def test_buying_back_cheaper_realizes_a_gain(self):
        c = make_contract(CALL, strike=110.0, expiry_day=60)
        h = EngineHarness(base_config(cash=100_000.0, margin=E.MarginModel.REG_T), [c])
        start = h.cash_micros
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=2.00)],
              groups=[group(sell(CALL, 1))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=2.00)],
              groups=[group(buy(CALL, 1, reduce_only=True))])
        h.bar(day_ns(3), [make_bar(CALL, timestamp_ns=day_ns(3), price=0.50)])
        assert h.cash_micros - start == round(1.50 * 100 * MICROS)


class TestLedgerIntegrity:
    def test_journal_sums_to_the_running_balance(self):
        c = make_contract(CALL, strike=100.0, expiry_day=60)
        h = EngineHarness(base_config(cash=25_000.0), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=4.44)],
              groups=[group(buy(CALL, 2))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=5.55)],
              groups=[group(sell(CALL, 1, reduce_only=True))])
        h.bar(day_ns(3), [make_bar(CALL, timestamp_ns=day_ns(3), price=5.55)])

        entries = h.engine.ledger_entries()
        assert sum(e.amount_micros for e in entries) == h.cash_micros
        assert h.engine.ledger_reconciles()

    def test_opening_deposit_is_the_first_entry(self):
        h = EngineHarness(base_config(cash=12_345.67),
                          [make_contract(CALL, strike=100.0, expiry_day=60)])
        first = h.engine.ledger_entries()[0]
        assert first.kind == E.LedgerEntryKind.DEPOSIT
        assert first.amount_micros == round(12_345.67 * MICROS)

    def test_every_fill_posts_a_premium_entry(self):
        c = make_contract(CALL, strike=100.0, expiry_day=60)
        h = EngineHarness(base_config(), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=3.00)],
              groups=[group(buy(CALL, 1))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=3.00)])
        premium = [e for e in h.engine.ledger_entries()
                   if e.kind == E.LedgerEntryKind.OPTION_PREMIUM]
        assert len(premium) == 1
        assert premium[0].amount_micros == -round(3.00 * 100 * MICROS)

    def test_ledger_reconciles_after_a_rejected_order(self):
        c = make_contract(CALL, strike=100.0, expiry_day=60)
        h = EngineHarness(base_config(cash=100.0), [c])
        before = h.cash_micros
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=50.00)],
              groups=[group(buy(CALL, 10))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=50.00)])
        assert h.rejections()
        assert h.cash_micros == before
        assert h.engine.ledger_reconciles()


class TestFeeArithmetic:
    def test_zero_schedule_charges_nothing(self):
        assert E.FeeSchedule.zero().option_fees(E.OrderSide.BUY, 10, 5_000.0) == 0.0

    def test_sell_side_carries_the_ad_valorem_fee(self):
        """Section 31 and the FINRA activity fee apply to sales only."""
        schedule = E.FeeSchedule()
        buy_fee = schedule.option_fees(E.OrderSide.BUY, 1, 1_000.0)
        sell_fee = schedule.option_fees(E.OrderSide.SELL, 1, 1_000.0)
        assert sell_fee > buy_fee

    def test_per_contract_fees_scale_linearly(self):
        schedule = E.FeeSchedule()
        one = schedule.option_fees(E.OrderSide.BUY, 1, 100.0)
        ten = schedule.option_fees(E.OrderSide.BUY, 10, 1_000.0)
        assert ten == pytest.approx(10 * one)

    def test_activity_fee_is_capped_per_trade(self):
        """
        The trading activity fee stops accruing at its per-trade cap, which the
        default $0.00329 rate reaches at 2,976 contracts.
        """
        schedule = E.FeeSchedule.zero()
        schedule.finra_taf_per_contract = 0.00329
        schedule.finra_taf_cap_per_trade = 9.79
        assert schedule.option_fees(E.OrderSide.SELL, 1_000, 0.0) == pytest.approx(3.29)
        assert schedule.option_fees(E.OrderSide.SELL, 100_000, 0.0) == pytest.approx(9.79)

    def test_sub_cent_charges_are_dropped_not_rounded_up(self):
        """A per-contract fee below a cent is not billed at all."""
        schedule = E.FeeSchedule.zero()
        schedule.cat_per_contract = 0.0003
        assert schedule.option_fees(E.OrderSide.BUY, 1, 0.0) == 0.0
        assert schedule.option_fees(E.OrderSide.BUY, 100, 0.0) == pytest.approx(0.03)

    def test_section_31_fee_rounds_up_to_the_cent(self):
        schedule = E.FeeSchedule.zero()
        schedule.sec_fee_rate_per_dollar = 0.0000206
        # $10,000 of proceeds owes $0.206, billed as $0.21.
        assert schedule.option_fees(E.OrderSide.SELL, 1, 10_000.0) == pytest.approx(0.21)

    def test_regulatory_fee_applies_to_both_sides(self):
        schedule = E.FeeSchedule.zero()
        schedule.regulatory_per_contract = 0.04
        assert schedule.option_fees(E.OrderSide.BUY, 5, 0.0) == pytest.approx(0.20)
        assert schedule.option_fees(E.OrderSide.SELL, 5, 0.0) == pytest.approx(0.20)

    def test_fees_reduce_cash_and_appear_in_the_journal(self):
        c = make_contract(CALL, strike=100.0, expiry_day=60)
        cfg = base_config(cash=50_000.0, fees=True)
        h = EngineHarness(cfg, [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=3.00)],
              groups=[group(buy(CALL, 1))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=3.00)])

        fee_entries = [e for e in h.engine.ledger_entries() if e.kind == E.LedgerEntryKind.FEE]
        assert len(fee_entries) == 1
        assert fee_entries[0].amount_micros < 0
        assert h.cash_micros == 50_000 * MICROS - round(3.00 * 100 * MICROS) + fee_entries[0].amount_micros
        assert h.engine.ledger_reconciles()

    def test_commission_per_contract_is_applied(self):
        schedule = E.FeeSchedule.zero()
        schedule.commission_per_contract = 0.65
        assert schedule.option_fees(E.OrderSide.BUY, 4, 0.0) == pytest.approx(2.60)


class TestEquityValuation:
    def test_equity_equals_cash_plus_position_value(self):
        c = make_contract(CALL, strike=100.0, expiry_day=60)
        h = EngineHarness(base_config(cash=20_000.0), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=5.00)],
              groups=[group(buy(CALL, 2))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=5.00)])
        state = h.bar(day_ns(3), [make_bar(CALL, timestamp_ns=day_ns(3), price=8.00)])
        assert state.cash_micros == 20_000 * MICROS - round(5.00 * 200 * MICROS)
        assert state.equity_micros == state.cash_micros + round(8.00 * 200 * MICROS)

    def test_unrealized_pnl_is_mark_minus_basis(self):
        c = make_contract(CALL, strike=100.0, expiry_day=60)
        h = EngineHarness(base_config(cash=20_000.0), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=5.00)],
              groups=[group(buy(CALL, 1))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=5.00)])
        state = h.bar(day_ns(3), [make_bar(CALL, timestamp_ns=day_ns(3), price=6.25)])
        assert state.unrealized_pnl == pytest.approx(1.25 * 100)


class TestRoundTripCostAttribution:
    """
    A TradeRecord is written on the CLOSING fill, so it used to record only the
    closing leg's fees and spread cost. The opening leg's costs were never
    attributed to the round trip, and the report's per-trade fee and spread
    figures came out at roughly half what the path actually paid: $5.58 against
    $10.02 of fees, $120.50 against $237.99 of spread cost, on the same run.
    """

    def _round_trip(self, *, contracts: int = 1, closes: tuple[int, ...] = (1,),
                    spread=E.SpreadModelKind.PROPORTIONAL_BPS, fees: bool = True,
                    expire: bool = False):
        contract = make_contract(CALL, strike=100.0, expiry_day=8)
        cfg = base_config(spread=spread, fees=fees)
        h = EngineHarness(cfg, [contract])
        day = 1
        h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=5.0)],
              groups=[group(buy(CALL, contracts))])
        day += 1
        for closing in closes:
            h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=6.0)],
                  groups=[group(sell(CALL, closing, reduce_only=True))])
            day += 1
        # One more bar so the final closing order fills.
        h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=6.0)])
        if expire:
            h.bar(day_ns(9), [make_bar(CALL, timestamp_ns=day_ns(9), price=6.0)])
            h.engine.end_session(day_ns(9) + 3_600_000_000_000)
        return h

    def test_trade_fees_account_for_both_legs(self):
        h = self._round_trip()
        trades = h.engine.trades()

        assert len(trades) == 1
        assert trades[0].fees_micros == h.finalize().fees_micros

    def test_trade_spread_cost_accounts_for_both_legs(self):
        h = self._round_trip()
        trades = h.engine.trades()

        assert trades[0].spread_cost_micros == h.finalize().spread_cost_micros

    def test_closing_only_one_leg_would_halve_the_figure(self):
        """
        Guards the specific defect: entry and exit are priced differently, so if
        only the exit were counted the totals could not match.
        """
        h = self._round_trip()
        fills = h.fills()

        assert len(fills) == 2
        assert fills[0].fees_micros > 0 and fills[1].fees_micros > 0
        assert h.engine.trades()[0].fees_micros == sum(f.fees_micros for f in fills)

    def test_partial_closes_sum_to_the_whole(self):
        """
        Entry costs release proportionally, the same way cost basis does, so three
        partial exits together carry exactly the one entry's costs.
        """
        h = self._round_trip(contracts=6, closes=(1, 2, 3))
        trades = h.engine.trades()
        metrics = h.finalize()

        assert len(trades) == 3
        assert h.quantity_of(CALL) == 0
        assert sum(t.fees_micros for t in trades) == metrics.fees_micros
        assert sum(t.spread_cost_micros for t in trades) == metrics.spread_cost_micros

    def test_an_expiring_position_still_carries_its_entry_costs(self):
        """
        A call bought and held to worthless expiry paid a fee and crossed a spread.
        The expiration path recorded zero for both.
        """
        contract = make_contract(CALL, strike=200.0, expiry_day=4)
        h = EngineHarness(base_config(spread=E.SpreadModelKind.PROPORTIONAL_BPS, fees=True),
                          [contract])
        for day in (1, 2):
            groups = [group(buy(CALL, 2))] if day == 1 else []
            h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=1.0)],
                  groups=groups)
        h.bar(day_ns(4), [make_bar(CALL, timestamp_ns=day_ns(4), price=0.01)],
              underlying={"TEST": 100.0})
        h.engine.end_session(day_ns(4) + 3_600_000_000_000)

        expired = [t for t in h.engine.trades() if t.reason == E.CloseReason.EXPIRED]
        assert len(expired) == 1
        assert expired[0].fees_micros == h.finalize().fees_micros > 0
        assert expired[0].spread_cost_micros == h.finalize().spread_cost_micros > 0

    def test_a_still_open_position_holds_its_entry_costs_back(self):
        """
        Entry costs are attributed when the round trip closes, not before, so an
        open position contributes nothing to trade statistics yet.
        """
        h = self._round_trip(contracts=4, closes=(1,))

        assert h.quantity_of(CALL) == 3
        trades = h.engine.trades()
        assert len(trades) == 1
        assert 0 < trades[0].fees_micros < h.finalize().fees_micros
