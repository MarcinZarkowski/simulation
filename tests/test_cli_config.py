"""
The command line's mapping onto engine config.

Every fail-closed gate the engine grew this session was unreachable from the CLI, so a
run could not turn one off even deliberately and, more importantly, nothing checked
that the flags mean what their names say. A flag wired to the wrong field is worse than
a missing one: it reports a setting the run did not apply.

The convention is `--allow-X` for a gate that defaults to refusing, so the default is
always the safe one and the flag is always the opt-out.
"""
from __future__ import annotations

import obt_engine as E
import pytest
from optionsbacktester import cli

REQUIRED = ["--strategy", "x", "--data-root", ".", "--tickers", "TEST"]


def config(*extra: str) -> E.BacktestConfig:
    """Parses through the CLI's own parser, so this cannot drift from it."""
    args = cli.parse_args(["run", *REQUIRED, *extra])
    cfg, _ = cli.build_config(args)
    return cfg


class TestTheDefaultsAreTheSafeOnes:
    def test_point_in_time_terms_are_required_by_default(self):
        assert config().require_point_in_time_terms is True

    def test_time_must_advance_by_default(self):
        assert config().require_monotonic_time is True

    def test_stale_bars_and_fallback_analytics_are_refused_by_default(self):
        cfg = config()

        assert cfg.reject_stale_bars is True
        assert cfg.reject_fallback_analytics is True

    def test_unconfirmed_lineage_is_refused_by_default(self):
        assert config().require_occ_confirmed_lineage is True

    def test_the_mark_age_bound_is_three_days_by_default(self):
        assert config().mark_age_limit_ns == 3 * 86_400 * 1_000_000_000

    def test_the_equity_curve_is_per_session_by_default(self):
        assert config().equity_curve_resolution == E.EquityCurveResolution.PER_SESSION

    def test_records_are_capped_by_default(self):
        assert config().max_retained_records == 1_000_000


class TestEachFlagTurnsOffExactlyItsOwnGate:
    """
    A flag wired to the wrong field is worse than a missing one, so each is checked
    against the field it names AND against the ones it must leave alone.
    """

    def test_allow_backfilled_terms(self):
        cfg = config("--allow-backfilled-terms")

        assert cfg.require_point_in_time_terms is False
        assert cfg.require_monotonic_time is True
        assert cfg.reject_stale_bars is True

    def test_allow_repeated_timestamps(self):
        cfg = config("--allow-repeated-timestamps")

        assert cfg.require_monotonic_time is False
        assert cfg.require_point_in_time_terms is True

    def test_allow_stale_bars(self):
        cfg = config("--allow-stale-bars")

        assert cfg.reject_stale_bars is False
        assert cfg.reject_fallback_analytics is True

    def test_allow_unconfirmed_lineage(self):
        cfg = config("--allow-unconfirmed-lineage")

        assert cfg.require_occ_confirmed_lineage is False


class TestTheNumericAndChoiceOptions:
    @pytest.mark.parametrize("days,expected_ns", [
        pytest.param(0.0, 0, id="zero_disables_the_bound"),
        pytest.param(1.0, 86_400 * 1_000_000_000, id="one_day"),
        pytest.param(0.5, 43_200 * 1_000_000_000, id="half_a_day"),
    ])
    def test_the_mark_age_limit_converts_days_to_nanoseconds(self, days, expected_ns):
        assert config("--mark-age-limit-days", str(days)).mark_age_limit_ns == expected_ns

    def test_the_equity_curve_can_be_set_per_bar(self):
        cfg = config("--equity-curve", "bar")

        assert cfg.equity_curve_resolution == E.EquityCurveResolution.PER_BAR

    def test_an_unbounded_record_cap_is_expressible(self):
        assert config("--max-retained-records", "0").max_retained_records == 0

    def test_the_variance_scale_reaches_the_spread_model(self):
        """The knob the report tells a reader to vary."""
        cfg = config("--spread-variance-scale", "2.5")

        assert cfg.spread_model.variance_scale == pytest.approx(2.5)


class TestEveryGateIsReachable:
    def test_no_fail_closed_gate_is_unreachable_from_the_command_line(self):
        """
        The gap this file closes. Each of these was added to the engine and left
        unreachable, so a run could not turn it off even deliberately.
        """
        defaults = config()
        overridden = config(
            "--allow-backfilled-terms", "--allow-repeated-timestamps",
            "--allow-stale-bars", "--allow-fallback-analytics",
            "--allow-unconfirmed-lineage", "--mark-age-limit-days", "0",
            "--equity-curve", "bar", "--max-retained-records", "0",
        )
        gates = ("require_point_in_time_terms", "require_monotonic_time",
                 "reject_stale_bars", "reject_fallback_analytics",
                 "require_occ_confirmed_lineage", "mark_age_limit_ns",
                 "max_retained_records")

        for gate in gates:
            assert getattr(defaults, gate) != getattr(overridden, gate), gate
        assert defaults.equity_curve_resolution != overridden.equity_curve_resolution
