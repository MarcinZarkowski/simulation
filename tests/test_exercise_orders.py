"""
Voluntary exercise.

``AssignmentPolicy::ExplicitExerciseOnly`` was selectable and a strategy had no way
to submit an exercise, so choosing it made the engine settle nothing at all: every
in-the-money position simply vanished at expiration with no cash flow.

Exercise is an order rather than a separate call so it inherits the group machinery
-- atomic with whatever replaces the position, and rejected through the same path
with the same named reasons.
"""
from __future__ import annotations

import obt_engine as E
import pytest
from optionsbacktester.strategy import buy, exercise, group, sell

from tests.conftest import EngineHarness, base_config, day_ns, make_bar, make_contract

CALL = 1
PUT = 2
STRIKE = 100.0
PREMIUM = 5.0
CASH = 100_000.0
SESSION = 3_600_000_000_000


def dollars(amount: float) -> int:
    return round(amount * 1_000_000)


def contracts(*, expiry_day: int = 60, **kw) -> list[E.OptionContractVersion]:
    return [
        make_contract(CALL, strike=STRIKE, expiry_day=expiry_day, **kw),
        make_contract(PUT, strike=STRIKE, expiry_day=expiry_day, is_call=False, **kw),
    ]


def holding(*, cv: int = CALL, quantity: int = 1, spot: float = 120.0,
            book: list | None = None, **cfg) -> EngineHarness:
    """Open `quantity` long contracts and return the harness holding them."""
    cfg.setdefault("margin", E.MarginModel.REG_T)
    h = EngineHarness(base_config(cash=CASH, **cfg), book or contracts())
    bars = lambda day: [make_bar(CALL, timestamp_ns=day_ns(day), price=PREMIUM),
                        make_bar(PUT, timestamp_ns=day_ns(day), price=PREMIUM)]
    for day in (1, 2):
        h.bar(day_ns(day), bars(day), groups=[group(buy(cv, quantity))] if day == 1 else [],
              underlying={"TEST": spot})
    return h


def _bare_snapshot(day: int) -> E.MarketSnapshot:
    """An option bar with no underlying price at all."""
    snap = E.MarketSnapshot()
    snap.timestamp = day_ns(day)
    snap.bars = [make_bar(CALL, timestamp_ns=day_ns(day), price=PREMIUM)]
    snap.underlying_price = {}
    return snap


def snapshot(day: int, spot: float, *, with_equity: bool = False) -> E.MarketSnapshot:
    snap = E.MarketSnapshot()
    snap.timestamp = day_ns(day)
    snap.bars = [make_bar(CALL, timestamp_ns=day_ns(day), price=PREMIUM),
                 make_bar(PUT, timestamp_ns=day_ns(day), price=PREMIUM)]
    snap.underlying_price = {"TEST": spot}
    if with_equity:
        bar = E.EquityBar()
        bar.timestamp = day_ns(day)
        bar.symbol = "TEST"
        bar.open = bar.high = bar.low = bar.close = spot
        bar.volume = 1_000_000
        snap.equity_bars = [bar]
    return snap


def submit_group(h: EngineHarness, order_group, *, day: int = 3, spot: float = 120.0,
                 with_equity: bool = False):
    """
    Submits on `day` and advances one bar so it fills.

    Execution is next-bar-open for an exercise as much as for anything else, so the
    settlement lands on the following bar.
    """
    for offset in (0, 1):
        h.engine.begin_bar(snapshot(day + offset, spot, with_equity=with_equity))
        if offset == 0:
            h.engine.submit_group(order_group)
        h.engine.end_bar()
    return h


def submit(h: EngineHarness, order, *, day: int = 3, spot: float = 120.0):
    return submit_group(h, group(order), day=day, spot=spot)


class TestExercisingACall:
    def test_the_contract_is_removed_and_shares_are_received(self):
        h = submit(holding(), exercise(CALL, 1))

        assert h.quantity_of(CALL) == 0
        assert h.shares_of("TEST") == 100

    def test_cash_moves_by_the_aggregate_exercise_price(self):
        h = submit(holding(), exercise(CALL, 1))

        # $500 premium paid, then $10,000 to take 100 shares at the $100 strike.
        assert h.cash_micros == dollars(CASH - 500.0 - 10_000.0)

    def test_the_trade_is_recorded_as_exercised(self):
        h = submit(holding(), exercise(CALL, 1))
        trades = [t for t in h.engine.trades() if t.contract_version_id == CALL]

        assert len(trades) == 1
        assert trades[0].reason == E.CloseReason.EXERCISED
        assert trades[0].quantity == 1

    def test_the_ledger_reconciles(self):
        h = submit(holding(), exercise(CALL, 1))

        assert h.engine.ledger_reconciles()

    def test_a_partial_exercise_leaves_the_remainder_open(self):
        h = submit(holding(quantity=5), exercise(CALL, 2))

        assert h.quantity_of(CALL) == 3
        assert h.shares_of("TEST") == 200

    def test_the_exercise_count_is_recorded(self):
        h = submit(holding(), exercise(CALL, 1))

        assert h.engine.finalize().exercise_count == 1


class TestExercisingAPut:
    def test_shares_are_delivered_and_the_strike_received(self):
        h = submit(holding(cv=PUT, spot=80.0), exercise(PUT, 1), spot=80.0)

        assert h.quantity_of(PUT) == 0
        assert h.shares_of("TEST") == -100        # delivered without owning
        assert h.cash_micros == dollars(CASH - 500.0 + 10_000.0)

    def test_a_put_exercised_against_owned_shares_flattens_them(self):
        """
        Delivery nets against an existing holding rather than opening a short
        alongside it.
        """
        from optionsbacktester.strategy import buy_shares

        h = holding(cv=PUT, spot=80.0)
        h = submit_group(h, group(buy_shares("TEST", 100)), spot=80.0, with_equity=True)
        assert h.shares_of("TEST") == 100

        h = submit_group(h, group(exercise(PUT, 1)), day=5, spot=80.0, with_equity=True)

        assert h.shares_of("TEST") == 0
        # Bought 100 at $80, delivered them at the $100 strike.
        assert h.cash_micros == dollars(CASH - 500.0 - 8_000.0 + 10_000.0)


class TestRefusals:
    def test_a_short_position_cannot_be_exercised(self):
        h = EngineHarness(base_config(cash=CASH, margin=E.MarginModel.REG_T), contracts())
        for day in (1, 2):
            h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=PREMIUM),
                                make_bar(PUT, timestamp_ns=day_ns(day), price=PREMIUM)],
                  groups=[group(sell(PUT, 1))] if day == 1 else [],
                  underlying={"TEST": 120.0})
        h = submit(h, exercise(PUT, 1))

        assert h.quantity_of(PUT) == -1
        assert h.rejections()[-1].reason == E.RejectReason.NOT_EXERCISABLE
        assert "only a long position" in h.rejections()[-1].detail

    def test_exercising_more_than_is_held_is_refused(self):
        h = submit(holding(quantity=2), exercise(CALL, 3))

        assert h.quantity_of(CALL) == 2
        assert "exceeds the position" in h.rejections()[-1].detail

    def test_a_position_that_does_not_exist_is_refused(self):
        h = submit(holding(cv=CALL), exercise(PUT, 1))

        assert h.rejections()[-1].reason == E.RejectReason.NOT_EXERCISABLE

    def test_a_european_contract_cannot_be_exercised_early(self):
        book = contracts(expiry_day=60)
        for c in book:
            c.is_american = False
        h = submit(holding(book=book), exercise(CALL, 1))

        assert h.quantity_of(CALL) == 1
        assert "European" in h.rejections()[-1].detail

    def test_an_american_contract_in_the_same_position_can_be(self):
        """The control: only the exercise style differs."""
        h = submit(holding(book=contracts(expiry_day=60)), exercise(CALL, 1))

        assert h.quantity_of(CALL) == 0

    def test_an_exercise_on_an_underlying_never_priced_is_refused(self):
        """
        Settling needs a spot. The engine has never observed one for this symbol, so
        there is nothing to compute a deliverable value against.
        """
        book = [make_contract(CALL, strike=STRIKE, expiry_day=60, underlying="NEVER"),
                make_contract(PUT, strike=STRIKE, expiry_day=60, is_call=False)]
        h = EngineHarness(base_config(cash=CASH, margin=E.MarginModel.REG_T), book)
        for day in (1, 2):
            snap = E.MarketSnapshot()
            snap.timestamp = day_ns(day)
            snap.bars = [make_bar(CALL, timestamp_ns=day_ns(day), price=PREMIUM)]
            snap.underlying_price = {}
            h.engine.begin_bar(snap)
            if day == 1:
                h.engine.submit_group(group(buy(CALL, 1)))
            h.engine.end_bar()
        assert h.quantity_of(CALL) == 1

        h.engine.begin_bar(_bare_snapshot(3))
        h.engine.submit_group(group(exercise(CALL, 1)))
        h.engine.end_bar()
        h.engine.begin_bar(_bare_snapshot(4))
        h.engine.end_bar()

        assert h.quantity_of(CALL) == 1
        assert h.rejections()[-1].reason == E.RejectReason.NO_MARKET_DATA


class TestOutOfTheMoneyExerciseIsAllowed:
    def test_a_holder_may_exercise_a_worthless_contract(self):
        """
        Legal, and occasionally rational for tax or delivery reasons. An engine that
        silently refuses it is deciding strategy rather than modelling a broker.
        """
        h = submit(holding(spot=80.0), exercise(CALL, 1), spot=80.0)

        assert h.quantity_of(CALL) == 0
        assert h.shares_of("TEST") == 100
        # Paid $10,000 for shares worth $8,000.
        assert h.cash_micros == dollars(CASH - 500.0 - 10_000.0)


class TestAtomicityWithOtherLegs:
    def test_an_exercise_can_be_grouped_with_a_share_sale(self):
        """
        Exercise-and-sell is one decision. Grouping it means the shares are never
        momentarily held unhedged.
        """
        from optionsbacktester.strategy import sell_shares

        h = submit_group(holding(), group(exercise(CALL, 1), sell_shares("TEST", 100)),
                         with_equity=True)

        assert h.quantity_of(CALL) == 0
        assert h.shares_of("TEST") == 0
        # Bought at $100 via exercise, sold at $120: +$2,000 less the $500 premium.
        assert h.cash_micros == dollars(CASH - 500.0 - 10_000.0 + 12_000.0)

    def test_a_refused_exercise_refuses_the_whole_group(self):
        """Atomicity applies to an exercise like any other leg."""
        h = submit_group(holding(quantity=1),
                         group(exercise(CALL, 5), sell(PUT, 1)))

        assert h.quantity_of(CALL) == 1
        assert h.quantity_of(PUT) == 0


class TestExplicitExerciseOnlyNowMeansSomething:
    """
    The policy was selectable and settled nothing, so an in-the-money position
    vanished at expiration with no cash flow at all.
    """

    def test_expiration_alone_settles_nothing_under_this_policy(self):
        h = holding(assignment=E.AssignmentPolicy.EXPLICIT_EXERCISE_ONLY,
                    book=contracts(expiry_day=5))
        h.bar(day_ns(5), [make_bar(CALL, timestamp_ns=day_ns(5), price=20.0)],
              underlying={"TEST": 120.0})
        h.engine.end_session(day_ns(5) + SESSION)

        assert h.quantity_of(CALL) == 1        # still held, nothing settled
        assert h.shares_of("TEST") == 0

    def test_an_explicit_exercise_settles_under_this_policy(self):
        h = holding(assignment=E.AssignmentPolicy.EXPLICIT_EXERCISE_ONLY,
                    book=contracts(expiry_day=8))
        h = submit(h, exercise(CALL, 1))

        assert h.quantity_of(CALL) == 0
        assert h.shares_of("TEST") == 100
        assert h.engine.ledger_reconciles()
