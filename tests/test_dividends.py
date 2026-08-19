"""
Cash dividends and dividend-driven early assignment.

``corporate_actions.parquet`` was loaded into ``DaySlice`` and never read, so a
dividend did nothing: it was never paid on a share position, and
``ConservativeEarlyAssignment`` -- declared as handling "the observable dividend
case" -- was byte-for-byte identical to ``AutomaticITMExercise``.
"""
from __future__ import annotations

import obt_engine as E
import pytest
from optionsbacktester.strategy import buy, group, sell

from tests.conftest import EngineHarness, base_config, day_ns, make_bar, make_contract

SPOT = 110.0
DIVIDEND = 1.00
EX_DAY = 10
PAY_DAY = 30
SESSION = 3_600_000_000_000


def dollars(amount: float) -> int:
    return round(amount * 1_000_000)


def dividend(*, amount: float = DIVIDEND, declared_day: int = 0,
             ex_day: int = EX_DAY, pay_day: int = PAY_DAY,
             symbol: str = "TEST") -> E.DividendEvent:
    d = E.DividendEvent()
    d.underlying_symbol = symbol
    d.amount_per_share = amount
    d.declared_at = day_ns(declared_day)
    d.ex_date = day_ns(ex_day)
    d.pay_date = day_ns(pay_day)
    return d


class DividendHarness:
    """
    Holds 100 shares from day 5, optionally with a short call written against
    them, and can be walked forward session by session.

    Shares arrive by exercising a deep-in-the-money long call, which is how the
    engine acquires equity at all: equity orders are not implemented.
    """

    SHARE_CALL = 1
    SHORT_CALL = 2

    def __init__(self, *, policy=E.AssignmentPolicy.AUTOMATIC_ITM_EXERCISE,
                 short_call_price: float | None = None, short_strike: float = 105.0,
                 dividends: list[E.DividendEvent] | None = None,
                 short_multiplier: int = 100, short_deliverable: float | None = None):
        share_call = make_contract(self.SHARE_CALL, strike=50.0, expiry_day=5)
        short_call = make_contract(self.SHORT_CALL, strike=short_strike, expiry_day=60,
                                   multiplier=short_multiplier,
                                   deliverable_shares=short_deliverable)
        self.short_call_price = short_call_price
        self.h = EngineHarness(base_config(assignment=policy), [share_call, short_call])
        if dividends:
            self.h.engine.queue_dividends(dividends)

        groups = [group(buy(self.SHARE_CALL, 1))]
        if short_call_price is not None:
            groups.append(group(sell(self.SHORT_CALL, 1)))
        self.session(1, groups=groups)
        for day in (2, 5):
            self.session(day)

    def _bars(self, day: int, spot: float, short_price: float | None):
        bars = [make_bar(self.SHARE_CALL, timestamp_ns=day_ns(day), price=max(spot - 50.0, 0.01))]
        price = self.short_call_price if short_price is None else short_price
        if price is not None:
            bars.append(make_bar(self.SHORT_CALL, timestamp_ns=day_ns(day), price=price))
        return bars

    def session(self, day: int, *, spot: float = SPOT, short_price: float | None = None,
                groups: list | None = None):
        self.h.bar(day_ns(day), self._bars(day, spot, short_price),
                   groups=groups or [], underlying={"TEST": spot})
        self.h.engine.end_session(day_ns(day) + SESSION)
        return self

    def walk(self, days, *, short_price: float | None = None):
        for day in days:
            self.session(day, short_price=short_price)
        return self

    @property
    def shares(self) -> int:
        return self.h.shares_of("TEST")

    def short_calls(self) -> int:
        return self.h.quantity_of(self.SHORT_CALL)

    def finalize(self):
        return self.h.finalize()

    def dividend_entries(self):
        return [e for e in self.h.engine.ledger_entries()
                if e.kind == E.LedgerEntryKind.DIVIDEND_CASH]


class TestDividendsArePaidOnShares:
    def test_shares_held_through_ex_date_earn_the_dividend(self):
        h = DividendHarness(dividends=[dividend()]).walk([9, EX_DAY, PAY_DAY, PAY_DAY + 1])

        assert h.shares == 100
        assert h.finalize().dividend_cash_micros == dollars(100.0)

    def test_no_dividend_data_pays_nothing(self):
        """The behaviour before corporate_actions was read: a whole yield missing."""
        h = DividendHarness().walk([9, EX_DAY, PAY_DAY, PAY_DAY + 1])

        assert h.shares == 100
        assert h.finalize().dividend_cash_micros == 0

    def test_cash_arrives_on_the_pay_date_not_the_ex_date(self):
        """
        Weeks separate the two, and accruing at ex-date overstates cash for that
        whole window -- which matters when the account is near a margin boundary.
        """
        h = DividendHarness(dividends=[dividend()]).walk([9, EX_DAY])
        assert h.dividend_entries() == []

        cash_before = h.h.cash_micros
        h.walk([PAY_DAY])

        assert len(h.dividend_entries()) == 1
        assert h.h.cash_micros - cash_before == dollars(100.0)

    def test_shares_acquired_after_ex_date_earn_nothing(self):
        """The shares arrive on day 5, so a day-3 ex-date passes over an empty book."""
        h = DividendHarness(dividends=[dividend(ex_day=3)]).walk([9, PAY_DAY, PAY_DAY + 1])

        assert h.shares == 100
        assert h.finalize().dividend_cash_micros == 0

    def test_short_shares_owe_the_dividend(self):
        """
        A naked short call assigned before ex-date leaves a short stock position,
        and the holder of that position pays the dividend rather than receiving it.
        """
        short_call = make_contract(1, strike=105.0, expiry_day=6)
        h = EngineHarness(base_config(margin=E.MarginModel.REG_T), [short_call])
        h.engine.queue_dividends([dividend()])
        for day in (1, 2, 6, 9, EX_DAY, PAY_DAY, PAY_DAY + 1):
            groups = [group(sell(1, 1))] if day == 1 else []
            h.bar(day_ns(day), [make_bar(1, timestamp_ns=day_ns(day), price=6.0)],
                  groups=groups, underlying={"TEST": SPOT})
            h.engine.end_session(day_ns(day) + SESSION)

        assert h.shares_of("TEST") == -100
        metrics = h.finalize()
        assert metrics.dividend_cash_micros == -dollars(100.0)
        assert metrics.ledger_reconciles

    def test_an_undeclared_dividend_is_not_accrued(self):
        """
        Accruing before the announcement would let a strategy collect cash from a
        payout that had not been declared yet.
        """
        h = DividendHarness(dividends=[dividend(declared_day=PAY_DAY + 5)])
        h.walk([9, EX_DAY, PAY_DAY, PAY_DAY + 1])

        assert h.finalize().dividend_cash_micros == 0

    def test_the_same_dividend_queued_twice_pays_once(self):
        h = DividendHarness(dividends=[dividend()])
        h.h.engine.queue_dividends([dividend(), dividend()])
        h.walk([9, EX_DAY, PAY_DAY, PAY_DAY + 1])

        assert h.finalize().dividend_cash_micros == dollars(100.0)

    def test_the_ledger_still_reconciles_exactly(self):
        h = DividendHarness(dividends=[dividend()]).walk([9, EX_DAY, PAY_DAY, PAY_DAY + 1])

        assert h.finalize().ledger_reconciles


class TestDividendDrivenEarlyAssignment:
    """
    A call holder exercises early to capture a dividend when the dividend exceeds
    the extrinsic value they give up. The short call is at 105 with the underlying
    at 110, so intrinsic is $500 and the mark sets the extrinsic.
    """

    EARLY = E.AssignmentPolicy.CONSERVATIVE_EARLY_ASSIGNMENT

    def _run(self, price: float, policy=EARLY, **kwargs):
        h = DividendHarness(policy=policy, short_call_price=price,
                            dividends=[dividend()], **kwargs)
        return h.walk([9, EX_DAY, PAY_DAY, PAY_DAY + 1])

    def test_extrinsic_below_the_dividend_assigns_the_short_call(self):
        h = self._run(5.20)   # extrinsic $20 < dividend $100

        assert h.short_calls() == 0
        assert h.shares == 0
        assert h.finalize().early_assignment_count == 1

    def test_extrinsic_above_the_dividend_does_not(self):
        h = self._run(6.50)   # extrinsic $150 > dividend $100

        assert h.short_calls() == -1
        assert h.shares == 100
        assert h.finalize().early_assignment_count == 0

    def test_indifference_holds_the_position(self):
        """Exactly equal is not a reason to exercise."""
        h = self._run(6.00)   # extrinsic $100 == dividend $100

        assert h.short_calls() == -1

    def test_an_out_of_the_money_call_is_never_assigned(self):
        h = self._run(0.50, short_strike=130.0)

        assert h.short_calls() == -1
        assert h.finalize().early_assignment_count == 0

    def test_the_default_policy_does_not_assign_early(self):
        """
        This is the defect: the two policies behaved identically, so a covered call
        kept a short leg the market would have taken away.
        """
        h = self._run(5.20, policy=E.AssignmentPolicy.AUTOMATIC_ITM_EXERCISE)

        assert h.short_calls() == -1
        assert h.finalize().early_assignment_count == 0

    def test_assignment_delivers_shares_at_the_strike(self):
        h = self._run(5.20)
        metrics = h.finalize()

        # Paid $6,000 for the share call, received $520, paid $5,000 on exercise,
        # delivered 100 shares at $105 for $10,500.
        assert h.h.cash_micros == dollars(100_000.0 - 6_000.0 + 520.0 - 5_000.0 + 10_500.0)
        assert metrics.dividend_cash_micros == 0    # no shares at ex-date
        assert metrics.ledger_reconciles

    def test_assignment_is_recorded_as_an_assignment_not_an_expiry(self):
        h = self._run(5.20)
        assigned = [t for t in h.h.engine.trades()
                    if t.contract_version_id == DividendHarness.SHORT_CALL]

        assert len(assigned) == 1
        assert assigned[0].reason == E.CloseReason.ASSIGNED
        assert assigned[0].was_short

    def test_the_decision_uses_the_prior_session_close_not_the_ex_date_open(self):
        """
        The underlying opens ex-date lower by roughly the dividend, so a call's
        extrinsic measured at the ex-date open is understated -- which would assign
        calls no rational holder would exercise.
        """
        h = DividendHarness(policy=self.EARLY, short_call_price=6.50,
                            dividends=[dividend()])
        h.session(9, short_price=6.50)              # pre-ex close: extrinsic $150
        h.session(EX_DAY, spot=109.0, short_price=5.50)   # ex-date: would read $50

        assert h.short_calls() == -1

    def test_a_long_call_is_never_assigned_early(self):
        """Assignment happens to a writer; a holder's own exercise is their choice."""
        share_call = make_contract(1, strike=50.0, expiry_day=60)
        h = EngineHarness(base_config(assignment=self.EARLY), [share_call])
        h.engine.queue_dividends([dividend()])
        for day in (1, 2, 9, EX_DAY):
            groups = [group(buy(1, 1))] if day == 1 else []
            h.bar(day_ns(day), [make_bar(1, timestamp_ns=day_ns(day), price=60.0)],
                  groups=groups, underlying={"TEST": SPOT})
            h.engine.end_session(day_ns(day) + SESSION)

        assert h.quantity_of(1) == 1
        assert h.finalize().early_assignment_count == 0

    @pytest.mark.parametrize("price,assigned", [
        pytest.param(5.20, True, id="extrinsic_10_below_dividend_50"),
        pytest.param(6.50, False, id="extrinsic_75_above_dividend_50"),
    ])
    def test_an_adjusted_deliverable_scales_both_sides(self, price, assigned):
        """
        A 50-share deliverable earns half the dividend and gives up half the
        extrinsic. Aggregate exercise price is 105 x 50 = $5,250 against $5,500
        delivered, so intrinsic is $250 and the dividend is $50; the mark sets the
        extrinsic at (price x 50) - 250.

        The two scale together, which is exactly why a real adjustment moves the
        multiplier with the deliverable: aggregate value is conserved, so the
        assignment decision is unchanged by the adjustment itself.
        """
        h = self._run(price, short_multiplier=50, short_deliverable=50.0)

        assert (h.short_calls() == 0) is assigned
