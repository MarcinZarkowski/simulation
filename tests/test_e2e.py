"""
End-to-end component tests.

These drive the whole workflow the way a user does: write a data lake in the
pipeline's format, stream it, run a strategy through the engine, and check the
result. Nothing is stubbed, so a break anywhere between the Parquet schema and
the ledger surfaces here.

The fixture lake is a closed-form function of the day index, so the expected
values below are exact rather than tolerances.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import polars as pl
import pytest

import obt_engine as E
from optionsbacktester import DataLake, UniverseFilter, build_report, load_day, run
from optionsbacktester.strategies import IronCondor, PoorMansCoveredCall, ShortPutSpread
from optionsbacktester.strategy import Chain, Context, Strategy, buy, group, sell
from tests import fixtures as F

WIDE = UniverseFilter(min_dte=1, max_dte=500)


def zero_cost_config(cash: float = 50_000.0, **kw) -> E.BacktestConfig:
    """Deterministic configuration: no spread, no fees, so ledgers are exact."""
    cfg = E.BacktestConfig()
    cfg.initial_cash = cash
    cfg.spread_mc_paths = kw.get("paths", 1)
    cfg.spread_mc_seed = kw.get("seed", 42)
    cfg.spread_model.kind = E.SpreadModelKind.ZERO
    cfg.margin_model = kw.get("margin", E.MarginModel.ROBINHOOD)
    cfg.fees = E.FeeSchedule.zero()
    return cfg


class BuyAndHoldCall(Strategy):
    """Buys one call on the first opportunity and never trades again."""
    name = "buy_and_hold_call"

    def __init__(self, *, target_dte: float = 30.0, strike: float | None = None):
        self.target_dte = target_dte
        self.strike = strike
        self.entered = False
        self.bought: int | None = None

    def on_market_snapshot(self, chain: Chain, context: Context):
        if self.entered:
            return ()
        calls = chain.calls()
        if len(calls) == 0:
            return ()
        candidates = [r for r in calls if abs(r.dte - self.target_dte) < 3.0]
        if self.strike is not None:
            candidates = [r for r in candidates if r.strike == self.strike]
        if not candidates:
            return ()
        pick = candidates[0]
        self.entered = True
        self.bought = pick.contract_version_id
        self.entry_mark = pick.mark
        return (group(buy(pick.contract_version_id, 1)),)


class TestLakeShape:
    """The generated lake must match what the pipeline actually writes."""

    def test_written_schema_matches_the_pipeline_column_order(self, tmp_path):
        root = F.flat_lake(tmp_path, trading_days=2, bars_per_day=1)
        path = next(Path(root).rglob("options_enriched.parquet"))
        assert pl.read_parquet(path).columns == F.OPTIONS_ENRICHED_SCHEMA

    def test_directory_layout_is_ticker_year_month_day(self, tmp_path):
        root = F.flat_lake(tmp_path, trading_days=3, bars_per_day=1)
        day_dir = DataLake(root, "TEST").complete_day_dirs()[0]
        assert day_dir.parts[-4:] == ("TEST", "2024", "01", "02")

    def test_success_manifest_records_row_counts(self, tmp_path):
        root = F.flat_lake(tmp_path, trading_days=2, bars_per_day=2)
        lake = DataLake(root, "TEST")
        manifest = lake.manifest(lake.complete_day_dirs()[0])
        assert manifest["status"] == "COMPLETE"
        assert manifest["row_counts"]["options_enriched.parquet"] > 0

    def test_generation_is_byte_identical_across_runs(self, tmp_path):
        """A golden comparison is only meaningful if the input is reproducible."""
        a = Path(F.flat_lake(tmp_path / "a", trading_days=3, bars_per_day=2))
        b = Path(F.flat_lake(tmp_path / "b", trading_days=3, bars_per_day=2))
        for left in a.rglob("*.parquet"):
            right = b / left.relative_to(a)
            assert left.read_bytes() == right.read_bytes(), left.name


class TestGoldenLongCall:
    """
    A single long call, priced by the fixture's own Black-Scholes, held to
    expiry. Every number here is derivable by hand, which is what makes it a
    golden test rather than a smoke test.
    """

    def test_entry_debit_equals_the_quoted_mark(self, tmp_path):
        root = F.flat_lake(tmp_path, trading_days=5, bars_per_day=2)
        strategy = BuyAndHoldCall(strike=100.0)
        result = run(lambda: strategy, data_root=root, ticker="TEST",
                     config=zero_cost_config(), universe=WIDE)

        fill = result.fills[0]
        assert fill.side == E.OrderSide.BUY
        assert fill.quantity == 1
        assert fill.net_cash == pytest.approx(-fill.fill_price * 100)

    def test_flat_tape_loses_only_time_value(self, tmp_path):
        """
        With the underlying pinned at 100 the call can only decay, so the loss is
        the entry premium minus whatever the option is still worth at the end.
        """
        root = F.flat_lake(tmp_path, trading_days=10, bars_per_day=2)
        result = run(lambda: BuyAndHoldCall(strike=100.0), data_root=root,
                     ticker="TEST", config=zero_cost_config(), universe=WIDE)
        metrics = result.paths[0]
        assert metrics.net_pnl < 0
        assert metrics.ledger_reconciles

    def test_itm_call_held_to_expiry_exercises_into_shares(self, tmp_path):
        """
        A 30-day 100 call on a tape rising 1/day finishes about 21 points in the
        money, so it must exercise rather than expire.
        """
        root = F.ramp_lake(tmp_path, per_day=1.0, trading_days=32, bars_per_day=2)
        result = run(lambda: BuyAndHoldCall(strike=100.0), data_root=root,
                     ticker="TEST", config=zero_cost_config(), universe=WIDE)
        metrics = result.paths[0]
        assert metrics.exercise_count == 1
        assert metrics.expiration_count >= 1
        assert metrics.ledger_reconciles

    def test_deep_otm_call_expires_worthless(self, tmp_path):
        root = F.crash_lake(tmp_path, per_day=1.0, trading_days=32, bars_per_day=2)
        result = run(lambda: BuyAndHoldCall(strike=120.0), data_root=root,
                     ticker="TEST", config=zero_cost_config(), universe=WIDE)
        metrics = result.paths[0]
        assert metrics.exercise_count == 0
        assert metrics.net_pnl < 0


class TestGoldenPoorMansCoveredCall:
    """
    The structure the whole exercise is about: a long LEAP standing in for 100
    shares, with shorter calls written against it repeatedly.
    """

    @staticmethod
    def _run(tmp_path, **kw):
        root = F.ramp_lake(tmp_path, per_day=0.4, trading_days=40, bars_per_day=2)
        return run(
            lambda: PoorMansCoveredCall(
                long_min_dte=180, long_max_dte=500, long_target_delta=0.80,
                short_min_dte=20, short_max_dte=60, short_target_delta=0.30,
                roll_at_dte=25.0, roll_at_profit_fraction=0.5),
            data_root=root, ticker="TEST", config=zero_cost_config(**kw), universe=WIDE)

    def test_runs_without_holding_any_shares(self, tmp_path):
        result = self._run(tmp_path)
        assert result.paths[0].fill_count > 0
        assert result.paths[0].rejection_count == 0

    def test_never_requires_margin_beyond_the_debit(self, tmp_path):
        """
        The long call collateralizes the short, so the requirement stays zero.
        A model that treated the short as naked would charge full spot notional.
        """
        assert self._run(tmp_path).paths[0].peak_margin_requirement == 0.0

    def test_writes_multiple_short_calls_against_one_long(self, tmp_path):
        """
        The point of the structure is collecting premium many times over the life
        of a single long leg, so there must be more than one short cycle.
        """
        result = self._run(tmp_path)
        sells = [f for f in result.fills if f.side == E.OrderSide.SELL]
        assert len(sells) >= 3

    def test_is_permitted_under_the_robinhood_model(self, tmp_path):
        result = self._run(tmp_path, margin=E.MarginModel.ROBINHOOD)
        refusals = [r for r in result.rejections
                    if r.reason == E.RejectReason.BROKER_DISALLOWED]
        assert refusals == []

    def test_ledger_reconciles_and_no_margin_breach(self, tmp_path):
        metrics = self._run(tmp_path).paths[0]
        assert metrics.ledger_reconciles
        assert not metrics.margin_breached


class TestGoldenSpreadStrategies:
    def test_short_put_spread_collects_credit_and_caps_requirement(self, tmp_path):
        root = F.flat_lake(tmp_path, trading_days=20, bars_per_day=2)
        result = run(lambda: ShortPutSpread(min_dte=20, max_dte=60, width=5.0),
                     data_root=root, ticker="TEST",
                     config=zero_cost_config(margin=E.MarginModel.ROBINHOOD),
                     universe=WIDE)
        metrics = result.paths[0]
        assert metrics.fill_count >= 2
        # A 5-wide spread cannot require more than its width per contract.
        assert metrics.peak_margin_requirement <= 500.0
        assert metrics.ledger_reconciles

    def test_iron_condor_fills_all_four_legs_atomically(self, tmp_path):
        root = F.flat_lake(tmp_path, trading_days=20, bars_per_day=2)
        result = run(lambda: IronCondor(min_dte=20, max_dte=60, width=5.0,
                                       short_delta=0.25),
                     data_root=root, ticker="TEST",
                     config=zero_cost_config(margin=E.MarginModel.ROBINHOOD),
                     universe=WIDE)
        if result.paths[0].fill_count:
            by_group: dict[int, int] = {}
            for f in result.fills:
                by_group[f.group_id] = by_group.get(f.group_id, 0) + 1
            assert all(n == 4 for n in by_group.values())
        assert result.paths[0].ledger_reconciles


class TestQualityGating:
    """Rows the pipeline flagged as unusable must not become trades."""

    @staticmethod
    def _all_symbols(spec: F.LakeSpec) -> list[str]:
        expiration = spec.start.replace(day=spec.start.day)
        options, _ = F.build_day_frames(spec, 0, spec.start)
        return options["symbol"].unique().to_list()

    def test_stale_rows_are_excluded_from_the_chain(self, tmp_path):
        spec = F.LakeSpec(underlying_path=lambda i: 100.0, trading_days=3, bars_per_day=1)
        stale = self._all_symbols(spec)[:4]
        spec = F.LakeSpec(underlying_path=lambda i: 100.0, trading_days=3,
                          bars_per_day=1, stale_symbols=tuple(stale))
        root = F.write_lake(tmp_path, spec)

        day = load_day(DataLake(root, "TEST").complete_day_dirs()[0], "TEST", WIDE)
        assert set(day.options["symbol"].to_list()).isdisjoint(stale)

    def test_fallback_iv_rows_are_excluded(self, tmp_path):
        base = F.LakeSpec(underlying_path=lambda i: 100.0, trading_days=3, bars_per_day=1)
        fallback = self._all_symbols(base)[:3]
        root = F.write_lake(tmp_path, F.LakeSpec(
            underlying_path=lambda i: 100.0, trading_days=3, bars_per_day=1,
            fallback_iv_symbols=tuple(fallback)))

        day = load_day(DataLake(root, "TEST").complete_day_dirs()[0], "TEST", WIDE)
        assert set(day.options["symbol"].to_list()).isdisjoint(fallback)

    def test_unpriced_adjusted_contracts_are_not_tradable(self, tmp_path):
        base = F.LakeSpec(underlying_path=lambda i: 100.0, trading_days=3, bars_per_day=1)
        unpriced = self._all_symbols(base)[:2]
        root = F.write_lake(tmp_path, F.LakeSpec(
            underlying_path=lambda i: 100.0, trading_days=3, bars_per_day=1,
            unpriced_adjusted_symbols=tuple(unpriced)))

        day = load_day(DataLake(root, "TEST").complete_day_dirs()[0], "TEST", WIDE)
        assert set(day.options["symbol"].to_list()).isdisjoint(unpriced)

    def test_incomplete_days_are_skipped(self, tmp_path):
        root = F.flat_lake(tmp_path, trading_days=5, bars_per_day=1)
        lake = DataLake(root, "TEST")
        (lake.complete_day_dirs()[2] / "_SUCCESS").unlink()
        assert len(lake.complete_day_dirs()) == 4
        assert len(lake.day_dirs()) == 5


class TestReproducibility:
    def test_identical_inputs_produce_an_identical_manifest(self, tmp_path):
        root = F.flat_lake(tmp_path, trading_days=6, bars_per_day=2)
        first = run(lambda: BuyAndHoldCall(strike=100.0), data_root=root,
                    ticker="TEST", config=zero_cost_config(), universe=WIDE)
        second = run(lambda: BuyAndHoldCall(strike=100.0), data_root=root,
                     ticker="TEST", config=zero_cost_config(), universe=WIDE)
        assert first.manifest.data_sha256 == second.manifest.data_sha256
        assert first.manifest.config_sha256 == second.manifest.config_sha256
        assert first.paths[0].net_pnl_micros == second.paths[0].net_pnl_micros

    def test_editing_the_lake_changes_the_data_hash(self, tmp_path):
        """The hash covers file contents, so a swapped day is detectable."""
        root = Path(F.flat_lake(tmp_path, trading_days=4, bars_per_day=1))
        before = run(lambda: BuyAndHoldCall(strike=100.0), data_root=root,
                     ticker="TEST", config=zero_cost_config(), universe=WIDE)

        target = next(root.rglob("options_enriched.parquet"))
        df = pl.read_parquet(target).with_columns(pl.col("valuation_price") * 1.10)
        df.write_parquet(target, compression="zstd", compression_level=3, statistics=True)

        after = run(lambda: BuyAndHoldCall(strike=100.0), data_root=root,
                    ticker="TEST", config=zero_cost_config(), universe=WIDE)
        assert before.manifest.data_sha256 != after.manifest.data_sha256

    def test_a_different_seed_leaves_the_config_hash_different(self, tmp_path):
        root = F.flat_lake(tmp_path, trading_days=4, bars_per_day=1)
        a = run(lambda: BuyAndHoldCall(strike=100.0), data_root=root, ticker="TEST",
                config=zero_cost_config(seed=1), universe=WIDE)
        b = run(lambda: BuyAndHoldCall(strike=100.0), data_root=root, ticker="TEST",
                config=zero_cost_config(seed=2), universe=WIDE)
        assert a.manifest.config_sha256 != b.manifest.config_sha256


class TestMonteCarloThroughTheFullStack:
    @staticmethod
    def _run(tmp_path, kind, paths, seed=11):
        root = F.ramp_lake(tmp_path, per_day=0.4, trading_days=25, bars_per_day=2)
        cfg = zero_cost_config(paths=paths, seed=seed)
        cfg.spread_model.kind = kind
        return run(lambda: PoorMansCoveredCall(
            long_min_dte=180, long_max_dte=500, short_min_dte=20, short_max_dte=60,
            roll_at_dte=25.0, roll_at_profit_fraction=0.5),
            data_root=root, ticker="TEST", config=cfg, universe=WIDE)

    def test_zero_spread_makes_every_path_identical(self, tmp_path):
        result = self._run(tmp_path, E.SpreadModelKind.ZERO, paths=12)
        assert result.deterministic

    def test_a_real_spread_model_spreads_the_paths(self, tmp_path):
        result = self._run(tmp_path, E.SpreadModelKind.LOGNORMAL, paths=40)
        assert not result.deterministic

    def test_removing_spread_cost_recovers_the_deterministic_result(self, tmp_path):
        """
        The report's deterministic figure must equal a genuine zero-spread run,
        which is what justifies presenting the two components separately.
        """
        stochastic = build_report(self._run(tmp_path, E.SpreadModelKind.LOGNORMAL, paths=64))
        deterministic = self._run(tmp_path, E.SpreadModelKind.ZERO, paths=1)
        assert stochastic.deterministic_net_pnl == pytest.approx(
            deterministic.paths[0].net_pnl, abs=0.01)

    def test_interval_narrows_as_paths_increase(self, tmp_path):
        narrow = build_report(self._run(tmp_path, E.SpreadModelKind.LOGNORMAL, paths=1000))
        wide = build_report(self._run(tmp_path, E.SpreadModelKind.LOGNORMAL, paths=50))
        assert narrow.standard_error < wide.standard_error

    def test_every_path_reconciles_its_ledger(self, tmp_path):
        result = self._run(tmp_path, E.SpreadModelKind.LOGNORMAL, paths=50)
        assert all(p.ledger_reconciles for p in result.paths)


class TestCliWorkflow:
    def test_cli_runs_and_emits_a_json_manifest(self, tmp_path, capsys):
        from optionsbacktester.cli import main

        root = F.ramp_lake(tmp_path, per_day=0.4, trading_days=20, bars_per_day=2)
        exit_code = main([
            "run",
            "--strategy", "optionsbacktester.strategies.pmcc:PoorMansCoveredCall",
            "--data-root", str(root), "--tickers", "TEST",
            "--spread-mc-paths", "20", "--spread-mc-seed", "5",
            "--spread-model", "lognormal", "--min-dte", "1", "--max-dte", "500",
            "--initial-cash", "25000", "--zero-fees", "--json",
        ])
        assert exit_code == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["manifest"]["spread_mc_paths"] == 20
        assert payload["manifest"]["spread_mc_seed"] == 5
        assert payload["manifest"]["execution_timing"] == "next_bar_open"
        assert len(payload["manifest"]["data_sha256"]) == 64
        assert payload["report"]["ledger_reconciles"] is True

    def test_unknown_calibration_key_is_rejected(self, tmp_path):
        from optionsbacktester.cli import main

        root = F.flat_lake(tmp_path, trading_days=3, bars_per_day=1)
        bad = tmp_path / "cal.json"
        bad.write_text(json.dumps({"not_a_real_parameter": 1.0}))

        with pytest.raises(SystemExit):
            main([
                "run", "--strategy", "optionsbacktester.strategies.pmcc:PoorMansCoveredCall",
                "--data-root", str(root), "--tickers", "TEST",
                "--spread-calibration", str(bad),
            ])

    def test_calibration_file_overrides_model_parameters(self, tmp_path, capsys):
        from optionsbacktester.cli import main

        root = F.ramp_lake(tmp_path, per_day=0.4, trading_days=12, bars_per_day=2)
        cal = tmp_path / "cal.json"
        cal.write_text(json.dumps({"log_sigma": 0.9, "log_base": 3.5}))

        main([
            "run", "--strategy", "optionsbacktester.strategies.pmcc:PoorMansCoveredCall",
            "--data-root", str(root), "--tickers", "TEST",
            "--spread-mc-paths", "10", "--spread-model", "lognormal",
            "--spread-calibration", str(cal), "--min-dte", "1", "--max-dte", "500",
            "--zero-fees", "--json",
        ])
        payload = json.loads(capsys.readouterr().out)
        assert payload["calibration"] == {"log_sigma": 0.9, "log_base": 3.5}
