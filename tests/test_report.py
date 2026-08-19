"""
Spread-model defaults, the variance control, the trade ledger, and the report.

The defaults get their own tests deliberately. Every other Monte Carlo test
overrides the spread model and zeroes the minimum-spread floor, which is exactly
how a degenerate default distribution went unnoticed: 400 draws at a $2.00 mark
produced one distinct value while the report printed a zero-width interval.
"""
from __future__ import annotations

import math
import statistics
import tempfile
from pathlib import Path

import pytest

import obt_engine as E
from optionsbacktester import (
    UniverseFilter,
    account_stats,
    build_performance_report,
    build_report,
    histogram,
    run,
    sparkline,
    trade_stats,
)
from optionsbacktester.strategy import buy, group, sell
from tests import fixtures as F
from tests.conftest import (
    EngineHarness,
    base_config,
    day_ns,
    make_bar,
    make_contract,
)

CALL = 1
REFERENCE_MARK = 10.0


def default_features(mark: float = 2.0) -> E.SpreadFeatures:
    """Features at the spread model's own documented reference point."""
    f = E.SpreadFeatures()
    f.mark_dollars = mark
    f.implied_volatility = 0.15
    f.days_to_expiry = 30.0
    f.volume = 5000.0
    return f


class TestShippedDefaults:
    """The configuration a user gets without overriding anything."""

    def test_default_model_is_conditional_lognormal(self):
        assert E.SpreadModelConfig().kind == E.SpreadModelKind.CONDITIONAL_LOGNORMAL

    def test_reference_point_median_matches_its_documented_magnitude(self):
        """
        log_base is documented as the log median spread in basis points at the
        reference point. It has to actually be that, or the constant is
        uninterpretable and the calibration is guesswork.
        """
        cfg = E.SpreadModelConfig()
        cfg.log_sigma = 0.0
        cfg.round_to_tick = False
        cfg.min_half_spread_cents = 0.0
        half = E.spread_draw(cfg, default_features(REFERENCE_MARK), 1, 0, 1, 1, 0, 0)
        full_bps = 2.0 * half / REFERENCE_MARK * 10_000
        assert full_bps == pytest.approx(math.exp(cfg.log_base), rel=0.01)

    @pytest.mark.parametrize("mark", [2.00, 5.00, 25.00])
    def test_defaults_produce_dispersion(self, mark):
        """
        More than one distinct draw, or the Monte Carlo says nothing while still
        printing a confidence interval.
        """
        cfg = E.SpreadModelConfig()
        draws = [E.spread_draw(cfg, default_features(mark), 42, s, 1, 1, 0, 0)
                 for s in range(400)]
        assert len(set(round(d, 8) for d in draws)) > 1
        assert statistics.stdev(draws) > 0.0

    @pytest.mark.parametrize("mark", [0.05, 1.00, 2.99, 3.00, 5.00, 50.00])
    def test_minimum_half_spread_is_never_violated(self, mark):
        """
        Tick rounding used to run after the floor, and half a cent rounds to zero
        on the five-cent grid, so any option at or above $3.00 filled at exactly
        the mark with no execution cost at all.
        """
        cfg = E.SpreadModelConfig()
        floor = cfg.min_half_spread_cents / 100.0
        for scenario in range(50):
            got = E.spread_draw(cfg, default_features(mark), 7, scenario, 1, 1, 0, 0)
            assert got >= floor - 1e-12, f"mark {mark} drew {got}, below floor {floor}"

    def test_a_cheap_option_is_pinned_at_the_tick_and_that_is_correct(self):
        """
        A $0.10 option genuinely cannot quote tighter than a penny, so a single
        deterministic value is the right answer here rather than a defect. The
        report is responsible for not presenting it as precision.
        """
        cfg = E.SpreadModelConfig()
        draws = {round(E.spread_draw(cfg, default_features(0.10), 3, s, 1, 1, 0, 0), 8)
                 for s in range(200)}
        assert len(draws) == 1


class TestVarianceScale:
    """The dispersion control, isolated from the level."""

    @staticmethod
    def _draws(scale: float, n: int = 4000) -> list[float]:
        cfg = E.SpreadModelConfig()
        cfg.kind = E.SpreadModelKind.LOGNORMAL
        cfg.round_to_tick = False
        cfg.min_half_spread_cents = 0.0
        cfg.variance_scale = scale
        f = default_features(5.0)
        return [E.spread_draw(cfg, f, 42, s, 1, 1, 0, 0) for s in range(n)]

    def test_zero_scale_collapses_every_draw_onto_one_value(self):
        assert len(set(round(d, 10) for d in self._draws(0.0, 200))) == 1

    def test_dispersion_grows_with_the_scale(self):
        low = statistics.stdev(self._draws(0.5))
        base = statistics.stdev(self._draws(1.0))
        high = statistics.stdev(self._draws(2.0))
        assert low < base < high

    def test_mean_is_preserved_across_scales(self):
        """
        Scaling sigma alone would also move a lognormal's mean, confounding level
        with dispersion. mu is compensated so the knob isolates variance.
        """
        means = [statistics.fmean(self._draws(s)) for s in (0.5, 1.0, 2.0)]
        assert max(means) / min(means) < 1.05

    def test_disabling_mean_preservation_lets_the_mean_move(self):
        cfg = E.SpreadModelConfig()
        cfg.kind = E.SpreadModelKind.LOGNORMAL
        cfg.round_to_tick = False
        cfg.min_half_spread_cents = 0.0
        cfg.preserve_mean_under_variance_scale = False
        f = default_features(5.0)

        def mean_at(scale: float) -> float:
            cfg.variance_scale = scale
            return statistics.fmean(
                E.spread_draw(cfg, f, 42, s, 1, 1, 0, 0) for s in range(4000))

        assert mean_at(3.0) > mean_at(1.0) * 1.2

    def test_effective_sigma_reflects_the_scale(self):
        cfg = E.SpreadModelConfig()
        cfg.log_sigma = 0.4
        cfg.variance_scale = 2.5
        assert cfg.effective_sigma() == pytest.approx(1.0)


class TestTradeLedger:
    """One record per closing event, with exact realized P&L."""

    @staticmethod
    def _round_trip(entry: float, exit_price: float, contracts: int = 1):
        c = make_contract(CALL, strike=100.0, expiry_day=400)
        h = EngineHarness(base_config(cash=100_000.0), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=entry)],
              groups=[group(buy(CALL, contracts))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=entry)],
              groups=[group(sell(CALL, contracts, reduce_only=True))])
        h.bar(day_ns(3), [make_bar(CALL, timestamp_ns=day_ns(3), price=exit_price,
                                   open_price=exit_price)])
        return h

    def test_a_closed_round_trip_emits_one_record(self):
        h = self._round_trip(2.00, 3.00)
        assert len(h.engine.trades()) == 1

    def test_realized_pnl_matches_the_price_difference(self):
        h = self._round_trip(2.00, 3.50)
        trade = h.engine.trades()[0]
        assert trade.realized_pnl == pytest.approx(1.50 * 100)
        assert trade.reason == E.CloseReason.CLOSED

    def test_entry_and_exit_prices_are_recorded_per_contract(self):
        h = self._round_trip(2.00, 3.50)
        trade = h.engine.trades()[0]
        assert trade.entry_price == pytest.approx(2.00)
        assert trade.exit_price == pytest.approx(3.50)

    def test_quantity_is_the_number_closed(self):
        assert self._round_trip(2.00, 3.00, contracts=3).engine.trades()[0].quantity == 3

    def test_a_partial_close_emits_a_record_for_the_closed_part(self):
        c = make_contract(CALL, strike=100.0, expiry_day=400)
        h = EngineHarness(base_config(cash=100_000.0), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=2.00)],
              groups=[group(buy(CALL, 5))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=2.00)],
              groups=[group(sell(CALL, 2, reduce_only=True))])
        h.bar(day_ns(3), [make_bar(CALL, timestamp_ns=day_ns(3), price=3.00,
                                   open_price=3.00)])
        trades = h.engine.trades()
        assert len(trades) == 1
        assert trades[0].quantity == 2
        assert h.quantity_of(CALL) == 3

    def test_an_expiring_position_is_recorded_with_its_reason(self):
        c = make_contract(CALL, strike=100.0, expiry_day=3)
        h = EngineHarness(base_config(cash=100_000.0), [c])
        h.bar(day_ns(1), [make_bar(CALL, timestamp_ns=day_ns(1), price=2.00)],
              groups=[group(buy(CALL, 1))])
        h.bar(day_ns(2), [make_bar(CALL, timestamp_ns=day_ns(2), price=2.00)])
        h.engine.end_session(day_ns(4))
        reasons = {t.reason for t in h.engine.trades()}
        assert reasons <= {E.CloseReason.EXPIRED, E.CloseReason.EXERCISED,
                           E.CloseReason.ASSIGNED}

    def test_metrics_aggregate_the_ledger(self):
        h = self._round_trip(2.00, 3.50)
        m = h.finalize()
        assert m.trade_count == 1
        assert m.winning_trades == 1
        assert m.losing_trades == 0
        assert m.best_trade_pnl == pytest.approx(150.0)

    def test_equity_points_decompose_realized_and_unrealized(self):
        h = self._round_trip(2.00, 3.50)
        points = h.engine.equity_points()
        assert len(points) == 3
        for p in points:
            assert p.equity == pytest.approx(p.cash + p.position_value)


class TestTradeStatistics:
    @staticmethod
    def _stats(values: list[float]):
        class FakeTrade:
            def __init__(self, pnl):
                self.realized_pnl = pnl
                self.fees = 0.0
                self.spread_cost = 0.0
                self.holding_days = 5
                self.reason = E.CloseReason.CLOSED
        return trade_stats([FakeTrade(v) for v in values])

    def test_empty_ledger_is_safe(self):
        s = self._stats([])
        assert s.count == 0
        assert s.z_histogram().total == 0

    def test_best_and_worst_are_the_extremes(self):
        s = self._stats([100.0, -50.0, 250.0, -300.0])
        assert s.best == 250.0
        assert s.worst == -300.0

    def test_median_is_the_middle_trade(self):
        assert self._stats([10.0, 20.0, 30.0]).median == 20.0

    def test_win_rate_excludes_scratches(self):
        """A zero-P&L close is neither a win nor a loss."""
        s = self._stats([100.0, -100.0, 0.0])
        assert (s.wins, s.losses, s.scratches) == (1, 1, 1)
        assert s.win_rate == pytest.approx(0.5)

    def test_profit_factor_is_gross_profit_over_gross_loss(self):
        s = self._stats([300.0, -100.0])
        assert s.profit_factor == pytest.approx(3.0)

    def test_profit_factor_is_infinite_with_no_losses(self):
        assert self._stats([10.0, 20.0]).profit_factor == float("inf")

    def test_streaks_are_counted(self):
        s = self._stats([1.0, 2.0, 3.0, -1.0, -2.0, 4.0])
        assert s.longest_win_streak == 3
        assert s.longest_loss_streak == 2

    def test_z_scores_are_standardized(self):
        s = self._stats([10.0, 20.0, 30.0, 40.0])
        assert statistics.fmean(s.z_scores) == pytest.approx(0.0, abs=1e-9)
        assert statistics.stdev(s.z_scores) == pytest.approx(1.0, rel=1e-6)

    def test_z_scores_are_zero_when_every_trade_is_identical(self):
        assert set(self._stats([5.0, 5.0, 5.0]).z_scores) == {0.0}

    def test_left_skew_is_detected(self):
        s = self._stats([10.0] * 9 + [-500.0])
        assert s.skew < -1.0

    def test_histogram_bins_sum_to_the_sample(self):
        h = self._stats([float(x) for x in range(50)]).pnl_histogram()
        assert h.total == 50

    def test_histogram_renders_without_a_display(self):
        rendered = self._stats([1.0, -1.0, 3.0]).z_histogram().render()
        assert "█" in rendered


class TestAccountStatistics:
    @staticmethod
    def _points(equities: list[float]):
        class FakePoint:
            def __init__(self, eq):
                self.equity = eq
                self.cash = eq
                self.realized_pnl = eq - 100.0
                self.unrealized_pnl = 0.0
                self.margin_requirement = 0.0
                self.position_value = 0.0
                self.open_positions = 1
                self.timestamp = 0
        return [FakePoint(e) for e in equities]

    def test_drawdown_is_measured_from_the_running_peak(self):
        a = account_stats(self._points([100.0, 150.0, 120.0, 160.0]), 100.0)
        assert a.max_drawdown == pytest.approx(30.0)
        assert a.max_drawdown_fraction == pytest.approx(0.2)

    def test_time_in_drawdown_is_a_fraction_of_the_run(self):
        """
        The first point sits at the starting peak, so it is not under water; the
        two dips are, and the final new high is not.
        """
        a = account_stats(self._points([100.0, 90.0, 90.0, 110.0]), 100.0)
        assert a.time_in_drawdown_fraction == pytest.approx(0.5)

    def test_monotonic_growth_has_no_drawdown(self):
        a = account_stats(self._points([100.0, 110.0, 120.0]), 100.0)
        assert a.max_drawdown == 0.0
        assert a.time_in_drawdown_fraction == 0.0

    def test_total_return_is_relative_to_starting_cash(self):
        a = account_stats(self._points([100.0, 125.0]), 100.0)
        assert a.total_return_fraction == pytest.approx(0.25)

    def test_empty_series_is_safe(self):
        assert account_stats([], 100.0).points == 0

    def test_sparkline_is_produced_without_a_display(self):
        assert len(sparkline([1.0, 5.0, 2.0, 8.0])) == 4


class TestDeterministicFigure:
    """
    'Before spread cost' has to come from a real zero-spread run.

    Adding mean spread cost back to the mean is only valid when the spread does
    not change which orders fill, and a limit or buying-power check can flip on
    the draw.
    """

    @staticmethod
    def _run(tmp_path, spread_kind, paths=8):
        root = F.ramp_lake(tmp_path, per_day=0.4, trading_days=25, bars_per_day=2)
        cfg = E.BacktestConfig()
        cfg.initial_cash = 30_000.0
        cfg.spread_mc_paths = paths
        cfg.spread_model.kind = spread_kind
        cfg.fees = E.FeeSchedule.zero()
        from optionsbacktester.strategies import PoorMansCoveredCall
        return run(lambda: PoorMansCoveredCall(
            long_min_dte=180, long_max_dte=500, short_min_dte=20, short_max_dte=60,
            roll_at_dte=25.0, roll_at_profit_fraction=0.5),
            data_root=root, ticker="TEST", config=cfg,
            universe=UniverseFilter(min_dte=1, max_dte=500), hash_data=False)

    def test_a_reference_run_supplies_the_figure(self, tmp_path):
        result = self._run(tmp_path, E.SpreadModelKind.LOGNORMAL)
        assert result.deterministic_pnl is not None

    def test_zero_spread_run_agrees_with_its_own_reference(self, tmp_path):
        result = self._run(tmp_path, E.SpreadModelKind.ZERO, paths=1)
        assert result.deterministic_pnl == pytest.approx(result.paths[0].net_pnl, abs=0.01)

    def test_report_reports_the_residual_rather_than_assuming_separability(self, tmp_path):
        report = build_report(self._run(tmp_path, E.SpreadModelKind.LOGNORMAL))
        implied = report.mean_net_pnl + report.mean_spread_cost
        assert report.separability_residual == pytest.approx(
            report.deterministic_net_pnl - implied, abs=1e-6)

    def test_degenerate_distribution_is_flagged(self, tmp_path):
        report = build_report(self._run(tmp_path, E.SpreadModelKind.ZERO, paths=6))
        assert report.degenerate
        assert "not reported" in report.summary()

    def test_a_real_distribution_is_not_flagged(self, tmp_path):
        report = build_report(self._run(tmp_path, E.SpreadModelKind.LOGNORMAL, paths=30))
        assert not report.degenerate
        assert "% interval" in report.summary()


class TestPerformanceReport:
    @staticmethod
    def _report(tmp_path):
        root = F.ramp_lake(tmp_path, per_day=0.4, trading_days=45, bars_per_day=2)
        cfg = E.BacktestConfig()
        cfg.initial_cash = 25_000.0
        cfg.spread_mc_paths = 20
        cfg.spread_model.kind = E.SpreadModelKind.LOGNORMAL
        from optionsbacktester.strategies import PoorMansCoveredCall
        result = run(lambda: PoorMansCoveredCall(
            long_min_dte=180, long_max_dte=500, short_min_dte=20, short_max_dte=60,
            roll_at_dte=25.0, roll_at_profit_fraction=0.5),
            data_root=root, ticker="TEST", config=cfg,
            universe=UniverseFilter(min_dte=1, max_dte=500), hash_data=False)
        return build_performance_report(result)

    def test_report_covers_account_trades_and_distribution(self, tmp_path):
        text = self._report(tmp_path).full()
        for heading in ("Account value", "Trades", "Monte Carlo",
                        "Trade P&L distribution"):
            assert heading in text

    def test_account_section_separates_realized_from_unrealized(self, tmp_path):
        text = self._report(tmp_path).account_summary()
        assert "of which realized" in text
        assert "of which unreal." in text

    def test_report_names_the_path_it_describes(self, tmp_path):
        report = self._report(tmp_path)
        assert 0 <= report.path_index < len(report.monte_carlo.percentiles) + 20

    def test_the_accounting_identity_closes_including_fees(self, tmp_path):
        """
        ending equity == start + realized + unrealized - fees.

        This is the invariant that actually constrains the ledger. The existing
        ledger_reconciles() check only asserts that the journal sums to the cash
        balance, which is true by construction and can never fail, so it cannot
        catch a realized-P&L error. Fees are posted to the journal separately from
        realized P&L, so they belong in the identity explicitly.
        """
        report = self._report(tmp_path)
        a = report.account
        fees = report.monte_carlo.mean_fees
        assert a.ending_equity == pytest.approx(
            a.starting_equity + a.ending_realized + a.ending_unrealized - fees,
            abs=0.02)


class TestSpreadAbsoluteValues:
    """
    Absolute-value oracles for every spread model.

    The audit's mutation run found exactly one survivor: doubling the
    `/ 20000.0` conversion changed every fill's slippage and all 396 tests still
    passed. The only ProportionalBps test asserted `rich == 10 * cheap`, and a
    ratio check cannot see a constant factor. These pin the constants themselves.
    """

    @staticmethod
    def _bare(kind) -> E.SpreadModelConfig:
        """A config with every guard off, so the model's own arithmetic is visible."""
        cfg = E.SpreadModelConfig()
        cfg.kind = kind
        cfg.round_to_tick = False
        cfg.min_half_spread_cents = 0.0
        cfg.max_fraction_of_mark = 1e9
        cfg.log_sigma = 0.0
        cfg.variance_scale = 0.0
        return cfg

    def test_constant_cents_is_half_the_configured_spread(self):
        cfg = self._bare(E.SpreadModelKind.CONSTANT_CENTS)
        cfg.constant_cents = 10.0
        got = E.spread_draw(cfg, default_features(5.0), 1, 0, 1, 1, 0, 0)
        assert got == pytest.approx(0.05)

    @pytest.mark.parametrize(("bps", "mark", "expected_half"), [
        (60.0, 5.00, 0.015),
        (100.0, 10.00, 0.050),
        (20.0, 1.00, 0.001),
        (250.0, 4.00, 0.050),
    ])
    def test_proportional_bps_converts_exactly(self, bps, mark, expected_half):
        """
        A full spread of `bps` basis points of the mark, halved:
        half = mark * bps / 20000. Pinning the constant, not just the ratio.
        """
        cfg = self._bare(E.SpreadModelKind.PROPORTIONAL_BPS)
        cfg.proportional_bps = bps
        got = E.spread_draw(cfg, default_features(mark), 1, 0, 1, 1, 0, 0)
        assert got == pytest.approx(expected_half, rel=1e-9)

    def test_lognormal_median_is_exp_log_base_in_basis_points(self):
        cfg = self._bare(E.SpreadModelKind.LOGNORMAL)
        cfg.log_base = math.log(80.0)
        got = E.spread_draw(cfg, default_features(10.0), 1, 0, 1, 1, 0, 0)
        assert got == pytest.approx(10.0 * 80.0 / 20000.0, rel=1e-9)

    def test_conditional_median_at_the_reference_point_is_exp_log_base(self):
        cfg = self._bare(E.SpreadModelKind.CONDITIONAL_LOGNORMAL)
        cfg.log_base = math.log(45.0)
        got = E.spread_draw(cfg, default_features(20.0), 1, 0, 1, 1, 0, 0)
        assert got == pytest.approx(20.0 * 45.0 / 20000.0, rel=1e-9)

    def test_empirical_returns_the_sampled_cents_exactly(self):
        cfg = self._bare(E.SpreadModelKind.EMPIRICAL)
        cfg.empirical_half_spread_cents = [2.5]
        got = E.spread_draw(cfg, default_features(5.0), 1, 0, 1, 1, 0, 0)
        assert got == pytest.approx(0.025)

    def test_zero_model_costs_nothing(self):
        cfg = self._bare(E.SpreadModelKind.ZERO)
        assert E.spread_draw(cfg, default_features(5.0), 1, 0, 1, 1, 0, 0) == 0.0

    def test_the_cap_binds_at_its_configured_fraction(self):
        cfg = self._bare(E.SpreadModelKind.PROPORTIONAL_BPS)
        cfg.proportional_bps = 100_000.0
        cfg.max_fraction_of_mark = 0.25
        got = E.spread_draw(cfg, default_features(8.0), 1, 0, 1, 1, 0, 0)
        assert got == pytest.approx(8.0 * 0.25)


class TestMoneyValidatesItsInput:
    """
    The only place the engine converts an unvalidated vendor float.

    Returning zero on a non-finite input is the worst available outcome: a NaN
    strike became strike 0, an always-in-the-money option whose cash-secured
    requirement is also zero.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_input_raises(self, bad):
        with pytest.raises(ValueError):
            E.Money.from_dollars(bad)

    @pytest.mark.parametrize("bad", [9.3e12, -9.3e12])
    def test_amounts_beyond_int64_microdollars_raise(self, bad):
        with pytest.raises((OverflowError, IndexError, ValueError)):
            E.Money.from_dollars(bad)

    def test_representable_amounts_still_convert(self):
        assert E.Money.from_dollars(1000.50).micros == 1_000_500_000
