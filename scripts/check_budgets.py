"""
Throughput and memory ceilings, checked rather than asserted in prose.

These are CEILINGS, not measurements. They exist so a regression that doubles peak
memory or halves throughput fails in CI instead of being discovered on a real run,
and they are set with enough headroom that ordinary machine-to-machine variation
does not trip them. A number here that is comfortably beaten is doing its job.

Run: uv run python scripts/check_budgets.py
"""
from __future__ import annotations

import math
import resource
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import obt_engine as E  # noqa: E402

from optionsbacktester import runner  # noqa: E402
from optionsbacktester.strategy import Strategy, buy, group, sell  # noqa: E402
from tests import fixtures as F  # noqa: E402

# Peak RSS for a 50-path run over a 40-day lake. The registry used to be copied per
# engine, which put this in the hundreds of megabytes at realistic contract counts.
MAX_PEAK_RSS_MIB = 700.0
# Rows per second through the streaming path, single path.
MIN_ROWS_PER_SECOND = 8_000.0
# How far throughput may fall going from 1 path to 50. Paths share one registry and
# one set of frames, so the marginal cost of a path is engine work alone.
MAX_SLOWDOWN_AT_50_PATHS = 6.0


def peak_rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


class Churn(Strategy):
    """Buys a 40-delta call and closes it next bar. Exercises the whole hot path."""

    name = "churn"

    def __init__(self):
        self.held = None

    def on_market_snapshot(self, chain, context):
        if self.held is not None:
            out = [group(sell(self.held, 1, reduce_only=True))]
            self.held = None
            return out
        pick = chain.calls().expiring_in(10, 70).nearest_delta(0.40)
        if pick is None:
            return ()
        self.held = pick.contract_version_id
        return [group(buy(self.held, 1))]


@dataclass
class Measurement:
    paths: int
    seconds: float
    rows: int
    fills: int

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.seconds if self.seconds else math.inf


def measure(root: Path, paths: int) -> Measurement:
    cfg = E.BacktestConfig()
    cfg.initial_cash = 100_000.0
    cfg.spread_mc_paths = paths
    cfg.spread_model.kind = E.SpreadModelKind.CONDITIONAL_LOGNORMAL
    started = time.perf_counter()
    result = runner.run(Churn, data_root=root, ticker="TEST", config=cfg,
                        hash_data=False)
    elapsed = time.perf_counter() - started
    return Measurement(paths, elapsed, result.manifest.option_row_count,
                       sum(p.fill_count for p in result.paths))


def main() -> int:
    root = Path(tempfile.mkdtemp())
    F.write_lake(root, F.LakeSpec(
        trading_days=40, bars_per_day=6,
        underlying_path=lambda i: 100.0 + 9.0 * math.sin(i / 2.4) + 0.25 * i,
    ))

    single = measure(root, 1)
    many = measure(root, 50)
    peak = peak_rss_mib()
    slowdown = many.seconds / single.seconds if single.seconds else math.inf

    failures: list[str] = []
    print(f"{'metric':<34}{'measured':>14}{'ceiling':>14}")
    for label, measured, limit, ok in (
        ("rows/second, 1 path", single.rows_per_second, MIN_ROWS_PER_SECOND,
         single.rows_per_second >= MIN_ROWS_PER_SECOND),
        ("slowdown, 1 -> 50 paths", slowdown, MAX_SLOWDOWN_AT_50_PATHS,
         slowdown <= MAX_SLOWDOWN_AT_50_PATHS),
        ("peak RSS (MiB)", peak, MAX_PEAK_RSS_MIB, peak <= MAX_PEAK_RSS_MIB),
    ):
        print(f"{label:<34}{measured:>14,.1f}{limit:>14,.1f}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(label)

    # A run that produced no fills would beat every ceiling by doing nothing.
    if single.fills == 0:
        failures.append("no fills: the benchmark did not exercise the engine")
        print("\nFAIL: the benchmark produced no fills, so the numbers above are empty.")

    if failures:
        print(f"\n{len(failures)} budget(s) exceeded: {', '.join(failures)}")
        return 1
    print(f"\nAll budgets met. {single.rows} rows, {single.fills} fills per path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
