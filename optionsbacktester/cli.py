"""
Command line runner.

Every choice that changes results is an explicit flag, and the flags that were
actually used are echoed into the run manifest. A result that cannot be tied back
to a seed, a spread model, and a data hash is not reproducible, so the manifest
is printed alongside the report rather than being optional.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import obt_engine as E

from .report import build_performance_report, build_report, convergence_table
from .runner import run
from .stream import UniverseFilter

SPREAD_MODELS = {
    "zero": E.SpreadModelKind.ZERO,
    "constant": E.SpreadModelKind.CONSTANT_CENTS,
    "proportional": E.SpreadModelKind.PROPORTIONAL_BPS,
    "lognormal": E.SpreadModelKind.LOGNORMAL,
    "conditional_lognormal": E.SpreadModelKind.CONDITIONAL_LOGNORMAL,
    "empirical": E.SpreadModelKind.EMPIRICAL,
}

EXECUTION_TIMING = {
    "next_bar_open": E.ExecutionTiming.NEXT_BAR_OPEN,
    "same_bar_close": E.ExecutionTiming.SAME_BAR_CLOSE,
}

ASSIGNMENT_POLICIES = {
    "expiration_only": E.AssignmentPolicy.EXPIRATION_ONLY,
    "explicit_exercise_only": E.AssignmentPolicy.EXPLICIT_EXERCISE_ONLY,
    "automatic_itm_exercise": E.AssignmentPolicy.AUTOMATIC_ITM_EXERCISE,
    "conservative_early_assignment": E.AssignmentPolicy.CONSERVATIVE_EARLY_ASSIGNMENT,
}

MARGIN_MODELS = {
    "cash": E.MarginModel.CASH_ACCOUNT,
    "reg_t": E.MarginModel.REG_T,
    "robinhood": E.MarginModel.ROBINHOOD,
    "portfolio": E.MarginModel.PORTFOLIO_APPROX,
}

# Path counts for the convergence check the spec requires before trusting a
# reported interval.
CONVERGENCE_PATHS = (100, 500, 1000, 5000)


def load_strategy(spec: str):
    """
    Resolve a strategy from a file path or a dotted module path.

    Accepts ``strategies/my_thing.py:MyThing``, ``strategies/my_thing.py`` (which
    takes the single Strategy subclass in the module), or
    ``optionsbacktester.strategies.pmcc:PoorMansCoveredCall``.
    """
    from .strategy import Strategy

    target, _, attr = spec.partition(":")
    if target.endswith(".py"):
        path = Path(target).resolve()
        if not path.exists():
            raise SystemExit(f"strategy file not found: {path}")
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)

    if attr:
        return getattr(module, attr)

    candidates = [
        obj for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy
        and not name.startswith("_")
    ]
    if len(candidates) != 1:
        raise SystemExit(
            f"{target} defines {len(candidates)} strategies; name one with 'path.py:ClassName'"
        )
    return candidates[0]


def apply_calibration(model: E.SpreadModelConfig, path: str | None) -> dict:
    """
    Overlay a JSON calibration onto the spread model.

    Returns what was applied so the manifest can record it. Unknown keys are an
    error rather than a silent no-op: a typo in a calibration file would
    otherwise leave the default in place and look like a valid run.
    """
    if not path:
        return {}
    payload = json.loads(Path(path).read_text())
    applied = {}
    for key, value in payload.items():
        if not hasattr(model, key):
            raise SystemExit(f"unknown spread-model parameter in {path}: {key}")
        setattr(model, key, value)
        applied[key] = value
    return applied


def build_config(args: argparse.Namespace) -> tuple[E.BacktestConfig, dict]:
    cfg = E.BacktestConfig()
    cfg.initial_cash = args.initial_cash
    cfg.spread_mc_paths = args.spread_mc_paths
    cfg.spread_mc_seed = args.spread_mc_seed
    cfg.spread_model.kind = SPREAD_MODELS[args.spread_model]
    cfg.spread_model.variance_scale = args.spread_variance_scale
    cfg.execution_timing = EXECUTION_TIMING[args.execution_timing]
    cfg.assignment_policy = ASSIGNMENT_POLICIES[args.assignment_policy]
    cfg.margin_model = MARGIN_MODELS[args.margin_model]
    cfg.require_occ_confirmed_lineage = not args.allow_unconfirmed_lineage
    cfg.reject_fallback_analytics = not args.allow_fallback_analytics
    cfg.reject_stale_bars = not args.allow_stale_bars
    if args.zero_fees:
        cfg.fees = E.FeeSchedule.zero()
    calibration = apply_calibration(cfg.spread_model, args.spread_calibration)
    return cfg, calibration


def build_universe(args: argparse.Namespace) -> UniverseFilter:
    return UniverseFilter(
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        min_abs_delta=args.min_abs_delta,
        max_abs_delta=args.max_abs_delta,
        min_volume=args.min_volume,
        exclude_stale=not args.allow_stale_bars,
        exclude_fallback_iv=not args.allow_fallback_analytics,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m optionsbacktester",
        description="Backtest an options strategy over the OptionsBackfill data lake",
    )
    p.add_argument("command", choices=["run", "convergence"],
                   help="run a backtest, or run one at several path counts to check "
                        "that the Monte Carlo error is falling as expected")
    p.add_argument("--strategy", required=True,
                   help="module:Class, or path/to/file.py[:Class]")
    p.add_argument("--data-root", required=True, help="root of the data lake")
    p.add_argument("--tickers", required=True, help="comma-separated underlyings")
    p.add_argument("--start", type=date.fromisoformat)
    p.add_argument("--end", type=date.fromisoformat)

    p.add_argument("--initial-cash", type=float, default=100_000.0)
    p.add_argument("--margin-model", choices=sorted(MARGIN_MODELS), default="robinhood")
    p.add_argument("--assignment-policy", choices=sorted(ASSIGNMENT_POLICIES),
                   default="automatic_itm_exercise")
    p.add_argument("--execution-timing", choices=sorted(EXECUTION_TIMING),
                   default="next_bar_open")

    p.add_argument("--spread-mc-paths", type=int, default=1000)
    p.add_argument("--spread-mc-seed", type=int, default=42)
    p.add_argument("--spread-model", choices=sorted(SPREAD_MODELS),
                   default="conditional_lognormal")
    p.add_argument("--spread-calibration", help="JSON file of spread-model parameters")
    p.add_argument("--spread-variance-scale", type=float, default=1.0,
                   help="scale the dispersion of the drawn bid/ask spread without "
                        "moving its mean. 1.0 is the model as calibrated, 2.0 doubles "
                        "the log-space standard deviation, 0.0 collapses every path "
                        "onto the mean. Use it to see how much of a result the spread "
                        "assumption owns.")
    p.add_argument("--report-confidence-level", type=float, default=0.95)

    p.add_argument("--min-dte", type=int)
    p.add_argument("--max-dte", type=int)
    p.add_argument("--min-abs-delta", type=float)
    p.add_argument("--max-abs-delta", type=float)
    p.add_argument("--min-volume", type=int)

    p.add_argument("--allow-stale-bars", action="store_true")
    p.add_argument("--allow-fallback-analytics", action="store_true")
    p.add_argument("--allow-unconfirmed-lineage", action="store_true",
                   help="carry positions through adjustments with no OCC-confirmed "
                        "lineage; unsafe, and off by default")
    p.add_argument("--zero-fees", action="store_true")
    p.add_argument("--allow-incomplete-days", action="store_true",
                   help="include days the pipeline did not mark complete")
    p.add_argument("--no-data-hash", action="store_true",
                   help="skip hashing the input files, which is faster but leaves the "
                        "run unverifiable")
    p.add_argument("--json", action="store_true", help="emit machine-readable output")
    return p.parse_args(argv)


def run_one(args: argparse.Namespace, ticker: str, paths: int | None = None):
    cfg, calibration = build_config(args)
    if paths is not None:
        cfg.spread_mc_paths = paths
    strategy_cls = load_strategy(args.strategy)
    result = run(
        strategy_cls,
        data_root=args.data_root,
        ticker=ticker,
        config=cfg,
        start=args.start,
        end=args.end,
        universe=build_universe(args),
        require_complete_days=not args.allow_incomplete_days,
        hash_data=not args.no_data_hash,
    )
    return result, calibration


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    for ticker in tickers:
        if args.command == "convergence":
            results = {n: run_one(args, ticker, n)[0] for n in CONVERGENCE_PATHS}
            print(f"\n=== {ticker} spread Monte Carlo convergence ===")
            print(convergence_table(results))
            continue

        result, calibration = run_one(args, ticker)
        if result.manifest.day_count == 0:
            print(f"{ticker}: no complete days found in range", file=sys.stderr)
            continue

        report = build_performance_report(
            result, args.report_confidence_level, args.spread_variance_scale)
        if args.json:
            print(json.dumps({
                "manifest": result.manifest.to_dict(),
                "calibration": calibration,
                "monte_carlo": dict(report.monte_carlo.__dict__),
                "account": {k: v for k, v in report.account.__dict__.items()
                            if not k.endswith("_curve")},
                "trades": {k: v for k, v in report.trades.__dict__.items()
                           if k not in ("z_scores", "pnl")},
                "path_index": report.path_index,
            }, indent=2, sort_keys=True, default=str))
            continue

        print(f"\n{'=' * 72}")
        print(f"  {ticker}")
        print("=" * 72)
        for key, value in result.manifest.to_dict().items():
            print(f"  {key:<20} {value}")
        if calibration:
            print(f"  {'calibration':<20} {calibration}")
        print(f"  {'variance_scale':<20} {args.spread_variance_scale:g}")
        print(f"  {'reported path':<20} median of {len(result.paths)} (index {report.path_index})")
        print()
        print(report.full())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
