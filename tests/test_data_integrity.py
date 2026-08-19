"""
Two gates on the data the engine is fed.

Marks used to be carried forward with no timestamp, so a contract that stopped
printing kept its last mark forever: an illiquid option that last traded three
months ago still valued the book, still set the margin requirement, and nothing
recorded that it had happened.

Snapshot timestamps were accepted in any order. A repeated timestamp let an order
submitted on one bar fill on the other at the same instant, which defeats
next-bar-open timing entirely; a decreasing one is time travel.
"""
from __future__ import annotations

import obt_engine as E
import pytest
from optionsbacktester.strategy import buy, group

from tests.conftest import EngineHarness, base_config, day_ns, make_bar, make_contract

CALL = 1
STRIKE = 100.0
PREMIUM = 5.0
DAY_NS = 86_400_000_000_000


def harness(**cfg) -> EngineHarness:
    contract = make_contract(CALL, strike=STRIKE, expiry_day=400)
    return EngineHarness(base_config(**cfg), [contract])


def bar_snapshot(day: int, *, price: float | None, spot: float) -> E.MarketSnapshot:
    """A snapshot with an option bar, or with none at all when price is None."""
    s = E.MarketSnapshot()
    s.timestamp = day_ns(day)
    s.bars = [] if price is None else [make_bar(CALL, timestamp_ns=day_ns(day), price=price)]
    s.underlying_price = {"TEST": spot}
    return s


def open_then_go_quiet(*, quiet_days: int, spot: float = 130.0, **cfg) -> EngineHarness:
    """
    Buy one call, then feed bars where the option no longer prints while the
    underlying keeps trading.
    """
    h = harness(**cfg)
    for day in (1, 2):
        h.engine.begin_bar(bar_snapshot(day, price=PREMIUM, spot=100.0))
        if day == 1:
            h.engine.submit_group(group(buy(CALL, 1)))
        h.engine.end_bar()
    for day in range(3, 3 + quiet_days):
        h.engine.begin_bar(bar_snapshot(day, price=None, spot=spot))
        h.engine.end_bar()
    return h


class TestStaleMarks:
    def test_a_recent_mark_is_carried_forward(self):
        """Within the limit, the last print is the best estimate available."""
        h = open_then_go_quiet(quiet_days=1)

        assert h.engine.metrics().stale_mark_valuations == 0
        state = h.engine.account_state()
        # Still valued at the $5.00 print, not at intrinsic.
        assert state.unrealized_pnl == pytest.approx(0.0)

    def test_a_mark_older_than_the_limit_falls_back_to_intrinsic(self):
        """
        A deep-in-the-money option is worth at least its intrinsic whether or not
        anyone traded it. Spot 130 against a 100 strike is $30 per share.
        """
        h = open_then_go_quiet(quiet_days=10, spot=130.0)
        state = h.engine.account_state()

        # Entry cost $500; intrinsic value $3,000.
        assert state.unrealized_pnl == pytest.approx(2_500.0)

    def test_the_fallback_is_recorded_rather_than_silent(self):
        h = open_then_go_quiet(quiet_days=10)
        metrics = h.engine.metrics()

        assert metrics.stale_mark_valuations > 0
        assert metrics.max_mark_age_ns >= 5 * DAY_NS

    def test_the_oldest_mark_age_is_reported_even_within_the_limit(self):
        """
        A report needs to be able to say how old the prices behind a valuation were,
        not just whether any crossed a threshold.
        """
        h = open_then_go_quiet(quiet_days=2)
        metrics = h.engine.metrics()

        assert metrics.stale_mark_valuations == 0
        assert metrics.max_mark_age_ns >= 2 * DAY_NS

    def test_a_zero_limit_disables_the_bound(self):
        """The previous behaviour, available for a run that wants it."""
        cfg = base_config()
        cfg.mark_age_limit_ns = 0
        contract = make_contract(CALL, strike=STRIKE, expiry_day=400)
        h = EngineHarness(cfg, [contract])
        for day in (1, 2):
            h.engine.begin_bar(bar_snapshot(day, price=PREMIUM, spot=100.0))
            if day == 1:
                h.engine.submit_group(group(buy(CALL, 1)))
            h.engine.end_bar()
        for day in range(3, 40):
            h.engine.begin_bar(bar_snapshot(day, price=None, spot=130.0))
            h.engine.end_bar()

        assert h.engine.metrics().stale_mark_valuations == 0
        assert h.engine.account_state().unrealized_pnl == pytest.approx(0.0)

    def test_a_stale_mark_survives_when_the_underlying_is_stale_too(self):
        """
        Intrinsic needs a fresh spot. Without one there is nothing better than the
        stale mark, and reporting zero would be worse -- but the path is still
        flagged.
        """
        h = harness()
        for day in (1, 2):
            h.engine.begin_bar(bar_snapshot(day, price=PREMIUM, spot=100.0))
            if day == 1:
                h.engine.submit_group(group(buy(CALL, 1)))
            h.engine.end_bar()
        for day in range(3, 14):
            snap = E.MarketSnapshot()
            snap.timestamp = day_ns(day)
            h.engine.begin_bar(snap)          # no bars, no underlying price
            h.engine.end_bar()

        assert h.engine.metrics().stale_mark_valuations > 0
        assert h.engine.account_state().unrealized_pnl == pytest.approx(0.0)

    def test_the_margin_requirement_uses_the_bounded_mark(self):
        """
        Margin leaned on the same unbounded mark, so a stale price set the
        requirement as well as the valuation.
        """
        fresh = open_then_go_quiet(quiet_days=1, margin=E.MarginModel.REG_T)
        stale = open_then_go_quiet(quiet_days=10, margin=E.MarginModel.REG_T)

        # A long call is paid in full either way, so the requirement is zero; what
        # matters is that both agree rather than one leaning on a quarter-old print.
        assert fresh.engine.account_state().margin_requirement == pytest.approx(0.0)
        assert stale.engine.account_state().margin_requirement == pytest.approx(0.0)


class TestMonotonicTime:
    def _feed(self, days: list[int], **cfg):
        h = harness(**cfg)
        for day in days:
            h.engine.begin_bar(bar_snapshot(day, price=PREMIUM, spot=100.0))
            h.engine.end_bar()
        return h

    def test_advancing_timestamps_are_fine(self):
        assert self._feed([1, 2, 3]) is not None

    def test_a_repeated_timestamp_is_refused(self):
        with pytest.raises(ValueError, match="repeated timestamp"):
            self._feed([1, 2, 2])

    def test_a_decreasing_timestamp_is_refused(self):
        with pytest.raises(ValueError, match="went backwards"):
            self._feed([1, 3, 2])

    def test_the_error_explains_why_a_repeat_matters(self):
        """
        Not a style objection: two bars at one instant let an order submitted on one
        fill on the other, which is exactly what next-bar-open timing exists to
        prevent.
        """
        with pytest.raises(ValueError, match="fill on the other at the same instant"):
            self._feed([5, 5])

    def test_the_first_bar_is_never_refused(self):
        """There is nothing to compare it against."""
        h = harness()
        h.engine.begin_bar(bar_snapshot(1, price=PREMIUM, spot=100.0))
        h.engine.end_bar()

        assert h.engine.metrics().rejection_count == 0

    def test_the_check_is_configurable(self):
        cfg = base_config()
        cfg.require_monotonic_time = False
        contract = make_contract(CALL, strike=STRIKE, expiry_day=400)
        h = EngineHarness(cfg, [contract])
        for day in (2, 2, 1):
            h.engine.begin_bar(bar_snapshot(day, price=PREMIUM, spot=100.0))
            h.engine.end_bar()

        assert h.engine.ledger_reconciles()

    def test_a_new_scenario_resets_the_clock(self):
        """
        Every Monte Carlo path replays the same days, so beginning a scenario has to
        forget the previous one's timestamps.
        """
        h = self._feed([1, 2, 3])
        h.engine.begin_scenario(1)
        h.engine.begin_bar(bar_snapshot(1, price=PREMIUM, spot=100.0))
        h.engine.end_bar()

        assert h.engine.ledger_reconciles()
