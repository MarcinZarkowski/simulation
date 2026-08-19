"""
Bounded record retention.

Per-path record vectors grew without limit. A ticker-year is 98,280 bars, so at
1,000 Monte Carlo paths the equity points alone came to 7.1 GB -- plus another
0.79 GB for `equity_curve_`, which stored exactly `EquityPoint.equity` a second time
and had no consumer at all.

The statistics are unaffected by the sampling: peak equity, max drawdown and peak
margin are running scalars updated on EVERY bar, so they stay exact at bar
resolution regardless of what the curve records. The curve is for shape.
"""
from __future__ import annotations

import obt_engine as E
import pytest
from optionsbacktester.analytics import account_stats
from optionsbacktester.strategy import buy, group, sell

from tests.conftest import EngineHarness, base_config, day_ns, make_bar, make_contract

CALL = 1
SESSION = 3_600_000_000_000


def harness(*, resolution=None, cap: int | None = None, **cfg) -> EngineHarness:
    contract = make_contract(CALL, strike=100.0, expiry_day=400)
    config = base_config(cash=1_000_000.0, **cfg)
    if resolution is not None:
        config.equity_curve_resolution = resolution
    if cap is not None:
        config.max_retained_records = cap
    return EngineHarness(config, [contract])


def walk(h: EngineHarness, bars: int, *, bars_per_session: int = 1,
         price=lambda i: 5.0 + (i % 7)):
    for i in range(1, bars + 1):
        h.bar(day_ns(i), [make_bar(CALL, timestamp_ns=day_ns(i), price=price(i))],
              underlying={"TEST": 100.0})
        if i % bars_per_session == 0:
            h.engine.end_session(day_ns(i) + SESSION)
    return h


class TestEquityCurveResolution:
    def test_per_session_records_one_point_per_session(self):
        h = walk(harness(resolution=E.EquityCurveResolution.PER_SESSION),
                 12, bars_per_session=4)

        assert len(h.engine.equity_points()) == 3

    def test_per_bar_records_one_point_per_bar(self):
        h = walk(harness(resolution=E.EquityCurveResolution.PER_BAR),
                 12, bars_per_session=4)

        assert len(h.engine.equity_points()) == 12

    def test_per_session_is_the_default(self):
        """
        390x smaller on a real feed, and it loses nothing the exact scalars already
        carry.
        """
        assert E.BacktestConfig().equity_curve_resolution \
            == E.EquityCurveResolution.PER_SESSION

    def test_a_session_point_carries_the_closing_state(self):
        """
        Not whichever bar happened to be written last -- the session's close.
        """
        h = walk(harness(resolution=E.EquityCurveResolution.PER_SESSION),
                 4, bars_per_session=4)
        point = h.engine.equity_points()[0]

        assert point.timestamp == day_ns(4)

    def test_finalize_flushes_the_last_pending_point(self):
        """
        A run whose caller never closes the final session must still have its ending
        state in the curve.
        """
        h = walk(harness(resolution=E.EquityCurveResolution.PER_SESSION),
                 3, bars_per_session=99)
        assert h.engine.equity_points() == []

        h.finalize()

        assert len(h.engine.equity_points()) == 1

    def test_beginning_a_scenario_clears_the_curve(self):
        h = walk(harness(resolution=E.EquityCurveResolution.PER_BAR), 5)
        assert h.engine.equity_points()

        h.engine.begin_scenario(1)

        assert h.engine.equity_points() == []


class TestTheExactStatisticsSurviveDownsampling:
    """
    The point of downsampling only being safe: what a reader is told about drawdown
    must not depend on how often the curve was sampled.
    """

    def _run(self, resolution):
        h = harness(resolution=resolution)
        # A sawtooth whose trough falls mid-session, so a per-session curve cannot
        # see it and a per-bar one can.
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=10.0)],
              groups=[group(buy(CALL, 10))], underlying={"TEST": 100.0})
        for day, price in ((2, 10.0), (3, 2.0), (4, 9.0)):
            h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=price)],
                  underlying={"TEST": 100.0})
        h.engine.end_session(day_ns(4) + SESSION)
        return h

    def test_the_exact_drawdown_is_identical_at_both_resolutions(self):
        coarse = self._run(E.EquityCurveResolution.PER_SESSION).finalize()
        fine = self._run(E.EquityCurveResolution.PER_BAR).finalize()

        assert coarse.max_drawdown == pytest.approx(fine.max_drawdown)
        assert coarse.peak_equity == pytest.approx(fine.peak_equity)

    def test_the_curve_alone_would_understate_it(self):
        """
        Which is why the report is given the exact figure rather than deriving one.
        """
        coarse = self._run(E.EquityCurveResolution.PER_SESSION)
        metrics = coarse.finalize()
        from_curve = account_stats(coarse.engine.equity_points(), 1_000_000.0)

        assert from_curve.max_drawdown < metrics.max_drawdown

    def test_the_report_uses_the_exact_figure(self):
        coarse = self._run(E.EquityCurveResolution.PER_SESSION)
        metrics = coarse.finalize()
        reported = account_stats(coarse.engine.equity_points(), 1_000_000.0, metrics)

        assert reported.max_drawdown == pytest.approx(metrics.max_drawdown)


class TestRecordRetentionCap:
    def _churn(self, cap: int, cycles: int) -> EngineHarness:
        h = harness(cap=cap)
        day = 1
        for _ in range(cycles):
            h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=5.0)],
                  groups=[group(buy(CALL, 1))], underlying={"TEST": 100.0})
            day += 1
            h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=5.0)],
                  groups=[group(sell(CALL, 1, reduce_only=True))], underlying={"TEST": 100.0})
            day += 1
        h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=5.0)],
              underlying={"TEST": 100.0})
        return h

    def test_fills_stop_accumulating_at_the_cap(self):
        h = self._churn(cap=3, cycles=5)

        assert len(h.fills()) == 3
        assert h.finalize().fill_count == 10       # the COUNT is still complete

    def test_the_dropped_records_are_counted(self):
        h = self._churn(cap=3, cycles=5)

        assert h.finalize().dropped_fills == 7

    def test_nothing_is_dropped_below_the_cap(self):
        h = self._churn(cap=1_000, cycles=5)
        metrics = h.finalize()

        assert len(h.fills()) == 10
        assert metrics.dropped_fills == 0
        assert metrics.dropped_trades == 0

    def test_a_zero_cap_means_unbounded(self):
        h = self._churn(cap=0, cycles=5)

        assert len(h.fills()) == 10
        assert h.finalize().dropped_fills == 0

    def test_the_ledger_still_reconciles_when_records_are_dropped(self):
        """
        Truncation affects the record LISTS, never the money. The ledger is the
        authority and it is not sampled.
        """
        h = self._churn(cap=2, cycles=6)
        metrics = h.finalize()

        assert metrics.dropped_fills > 0
        assert metrics.ledger_reconciles

    def test_trades_are_capped_too(self):
        h = self._churn(cap=2, cycles=5)
        metrics = h.finalize()

        assert len(h.engine.trades()) == 2
        assert metrics.trade_count == 5
        assert metrics.dropped_trades == 3

    def test_rejections_are_capped_too(self):
        h = harness(cap=2)
        for day in range(1, 7):
            # Exercising a position that does not exist is refused every time.
            order = E.Order()
            order.contract_version_id = CALL
            order.quantity = 1
            order.type = E.OrderType.EXERCISE
            h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=5.0)],
                  groups=[group(order)], underlying={"TEST": 100.0})
        metrics = h.finalize()

        assert len(h.rejections()) == 2
        assert metrics.rejection_count > 2
        assert metrics.dropped_rejections > 0
