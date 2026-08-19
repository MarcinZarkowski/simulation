"""
Cash-settled index options.

``is_american`` and the settlement style were declared on the contract and read
nowhere, so every contract settled by delivering shares. An SPX option would have
booked a share position in an index nobody can deliver, and a European contract
was eligible for the dividend-driven early assignment that only an American one
can receive.

OCC Rule 1804 sets the exercise-by-exception threshold for a cash-settled contract
at $1.00, which is what ``aggregate_exercise_threshold`` produces for a 100-unit
contract, so one expression covers both settlement styles.
"""
from __future__ import annotations

import obt_engine as E
import pytest
from optionsbacktester.strategy import buy, group, sell

from tests.conftest import EngineHarness, base_config, day_ns, make_bar, make_contract

SPX = "SPXINDEX"
STRIKE = 5_000.0
EXPIRY_DAY = 30
PREMIUM = 20.0
INITIAL_CASH = 1_000_000.0
SESSION = 3_600_000_000_000


def dollars(amount: float) -> int:
    return round(amount * 1_000_000)


def index_contract(cv_id: int = 1, *, strike: float = STRIKE, is_call: bool = True,
                   expiry_day: int = EXPIRY_DAY, last_trade_day: int | None = None):
    c = make_contract(cv_id, strike=strike, is_call=is_call, expiry_day=expiry_day,
                      underlying=SPX)
    c.settlement_style = E.SettlementStyle.CASH_SETTLEMENT
    c.is_american = False
    if last_trade_day is not None:
        c.last_trade_at = day_ns(last_trade_day)
    return c


def hold_to_expiry(contract, *, spot: float, settlement: float | None = None,
                   quantity: int = 1, cash: float = INITIAL_CASH):
    """Open a position, hold it to expiration, and settle."""
    cfg = base_config(cash=cash, margin=E.MarginModel.REG_T)
    h = EngineHarness(cfg, [contract])
    order = buy(contract.id, quantity) if quantity > 0 else sell(contract.id, -quantity)
    for day in (1, 2):
        h.bar(day_ns(day), [make_bar(contract.id, timestamp_ns=day_ns(day), price=PREMIUM)],
              groups=[group(order)] if day == 1 else [], underlying={SPX: spot})
    prices = {SPX: settlement} if settlement is not None else None
    h.engine.begin_bar(_snapshot(contract, EXPIRY_DAY, spot, prices))
    h.engine.end_bar()
    h.engine.end_session(day_ns(EXPIRY_DAY) + SESSION)
    return h


def _snapshot(contract, day, spot, settlement):
    s = E.MarketSnapshot()
    s.timestamp = day_ns(day)
    s.bars = [make_bar(contract.id, timestamp_ns=day_ns(day), price=PREMIUM)]
    s.underlying_price = {SPX: spot}
    if settlement:
        s.settlement_price = settlement
    return s


class TestCashSettlement:
    def test_an_itm_long_call_receives_the_intrinsic_in_cash(self):
        h = hold_to_expiry(index_contract(), spot=5_100.0)

        # Paid 20.00 x 100, received (5,100 - 5,000) x 100.
        assert h.cash_micros == dollars(INITIAL_CASH - 2_000.0 + 10_000.0)
        assert h.engine.ledger_reconciles()

    def test_no_share_position_is_created(self):
        """
        The defect. settle_physically would have booked 100 shares of an index,
        which nobody can deliver and which has no market to close into.
        """
        h = hold_to_expiry(index_contract(), spot=5_100.0)

        assert h.shares_of(SPX) == 0
        assert h.equity_positions() == []

    def test_an_itm_short_call_pays_the_intrinsic(self):
        h = hold_to_expiry(index_contract(), spot=5_100.0, quantity=-1)

        assert h.cash_micros == dollars(INITIAL_CASH + 2_000.0 - 10_000.0)
        assert h.shares_of(SPX) == 0

    def test_an_itm_long_put_receives_the_intrinsic(self):
        h = hold_to_expiry(index_contract(is_call=False), spot=4_900.0)

        assert h.cash_micros == dollars(INITIAL_CASH - 2_000.0 + 10_000.0)
        assert h.shares_of(SPX) == 0

    def test_an_otm_contract_expires_worthless(self):
        h = hold_to_expiry(index_contract(), spot=4_900.0)

        assert h.cash_micros == dollars(INITIAL_CASH - 2_000.0)
        assert h.shares_of(SPX) == 0

    def test_the_trade_is_recorded_as_exercised_not_expired(self):
        h = hold_to_expiry(index_contract(), spot=5_100.0)
        trades = h.engine.trades()

        assert len(trades) == 1
        assert trades[0].reason == E.CloseReason.EXERCISED

    def test_a_fractional_deliverable_does_not_block_cash_settlement(self):
        """
        There is no fraction of a share to owe cash in lieu of when the whole
        settlement is already cash.
        """
        c = index_contract()
        c.deliverable_equity_microshares = 100_500_000   # 100.5 units

        h = hold_to_expiry(c, spot=5_100.0)

        assert h.engine.finalize().quarantined_positions == 0
        assert h.engine.ledger_reconciles()


class TestOfficialSettlementValue:
    """
    SET for SPX and VRO for VIX are computed from opening prints and can differ
    materially from the last bar anyone observed.
    """

    def test_an_official_value_is_preferred_over_the_observed_spot(self):
        h = hold_to_expiry(index_contract(), spot=5_100.0, settlement=5_200.0)

        assert h.cash_micros == dollars(INITIAL_CASH - 2_000.0 + 20_000.0)

    def test_falling_back_to_the_observed_spot_is_counted(self):
        h = hold_to_expiry(index_contract(), spot=5_100.0)

        assert h.engine.finalize().settlements_without_official_price == 1

    def test_using_the_official_value_is_not_counted_as_a_fallback(self):
        h = hold_to_expiry(index_contract(), spot=5_100.0, settlement=5_200.0)

        assert h.engine.finalize().settlements_without_official_price == 0

    def test_an_official_value_alone_is_enough_to_settle(self):
        """
        An AM-settled series stops trading before expiration, so there may be no
        option bar and no observed spot on settlement morning -- only the published
        number.
        """
        c = index_contract()
        cfg = base_config(cash=INITIAL_CASH, margin=E.MarginModel.REG_T)
        h = EngineHarness(cfg, [c])
        for day in (1, 2):
            h.bar(day_ns(day), [make_bar(c.id, timestamp_ns=day_ns(day), price=PREMIUM)],
                  groups=[group(buy(c.id, 1))] if day == 1 else [],
                  underlying={SPX: 5_100.0})
        settlement_only = E.MarketSnapshot()
        settlement_only.timestamp = day_ns(EXPIRY_DAY)
        settlement_only.settlement_price = {SPX: 5_050.0}
        h.engine.begin_bar(settlement_only)
        h.engine.end_bar()

        assert h.engine.finalize().quarantined_positions == 0
        assert h.cash_micros == dollars(INITIAL_CASH - 2_000.0 + 5_000.0)


class TestEuropeanExercise:
    def test_a_european_contract_is_not_assigned_early(self):
        """
        A European contract cannot be exercised before expiration, so it cannot be
        assigned early either -- not conservatively, not at all.
        """
        c = index_contract(expiry_day=60)
        c.strike = 5_000.0
        cfg = base_config(cash=INITIAL_CASH, margin=E.MarginModel.REG_T,
                          assignment=E.AssignmentPolicy.CONSERVATIVE_EARLY_ASSIGNMENT)
        h = EngineHarness(cfg, [c])
        d = E.DividendEvent()
        d.underlying_symbol = SPX
        d.amount_per_share = 50.0        # enormous, to make the condition bite
        d.declared_at = day_ns(0)
        d.ex_date = day_ns(10)
        d.pay_date = day_ns(30)
        h.engine.queue_dividends([d])
        for day in (1, 2, 9, 10):
            h.bar(day_ns(day), [make_bar(c.id, timestamp_ns=day_ns(day), price=101.0)],
                  groups=[group(sell(c.id, 1))] if day == 1 else [],
                  underlying={SPX: 5_100.0})
            h.engine.end_session(day_ns(day) + SESSION)

        assert h.quantity_of(c.id) == -1
        assert h.engine.finalize().early_assignment_count == 0

    def test_an_american_contract_in_the_same_position_is_assigned(self):
        """The control: only the exercise style differs."""
        c = index_contract(expiry_day=60)
        c.is_american = True
        cfg = base_config(cash=INITIAL_CASH, margin=E.MarginModel.REG_T,
                          assignment=E.AssignmentPolicy.CONSERVATIVE_EARLY_ASSIGNMENT)
        h = EngineHarness(cfg, [c])
        d = E.DividendEvent()
        d.underlying_symbol = SPX
        d.amount_per_share = 50.0
        d.declared_at = day_ns(0)
        d.ex_date = day_ns(10)
        d.pay_date = day_ns(30)
        h.engine.queue_dividends([d])
        for day in (1, 2, 9, 10):
            h.bar(day_ns(day), [make_bar(c.id, timestamp_ns=day_ns(day), price=101.0)],
                  groups=[group(sell(c.id, 1))] if day == 1 else [],
                  underlying={SPX: 5_100.0})
            h.engine.end_session(day_ns(day) + SESSION)

        assert h.quantity_of(c.id) == 0
        assert h.engine.finalize().early_assignment_count == 1


class TestTradingStopsBeforeSettlement:
    """
    An AM-settled series stops trading the business day BEFORE expiration and
    settles against the next morning's opening prints. A bar-based feed that
    ignores the distinction lets a strategy trade a contract that no longer exists.
    """

    def _try_to_open(self, contract, day):
        cfg = base_config(cash=INITIAL_CASH, margin=E.MarginModel.REG_T)
        h = EngineHarness(cfg, [contract])
        for d in (day, day + 1):
            h.bar(day_ns(d), [make_bar(contract.id, timestamp_ns=day_ns(d), price=PREMIUM)],
                  groups=[group(buy(contract.id, 1))] if d == day else [],
                  underlying={SPX: 5_100.0})
        return h

    def test_a_new_position_is_refused_after_the_last_trade_instant(self):
        c = index_contract(expiry_day=30, last_trade_day=10)

        h = self._try_to_open(c, 15)

        assert h.quantity_of(c.id) == 0
        assert "stopped trading" in h.rejections()[0].detail

    def test_a_new_position_is_accepted_before_it(self):
        c = index_contract(expiry_day=30, last_trade_day=10)

        assert self._try_to_open(c, 5).quantity_of(c.id) == 1

    def test_the_boundary_is_inclusive_and_measured_at_fill_time(self):
        """
        The last trading day is a trading day -- but the gate is checked when the
        order FILLS, not when it is submitted. Execution is next-bar-open, so an
        order entered on the last trading day fills the morning after trading has
        stopped, and is correctly refused.
        """
        c = index_contract(expiry_day=30, last_trade_day=6)

        assert self._try_to_open(c, 5).quantity_of(c.id) == 1     # fills on day 6
        assert self._try_to_open(c, 6).quantity_of(c.id) == 0     # would fill on day 7

    def test_an_existing_position_can_still_be_closed(self):
        c = index_contract(expiry_day=30, last_trade_day=10)
        h = self._try_to_open(c, 5)
        assert h.quantity_of(c.id) == 1

        for day in (15, 16):
            h.bar(day_ns(day), [make_bar(c.id, timestamp_ns=day_ns(day), price=PREMIUM)],
                  groups=[group(sell(c.id, 1, reduce_only=True))] if day == 15 else [],
                  underlying={SPX: 5_100.0})

        assert h.quantity_of(c.id) == 0

    def test_an_unset_last_trade_instant_means_trading_through_expiration(self):
        """A PM-settled contract trades to the close on its expiration day."""
        c = index_contract(expiry_day=30)

        assert self._try_to_open(c, 20).quantity_of(c.id) == 1
