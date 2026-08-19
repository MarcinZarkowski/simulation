"""
Golden end-to-end replay.

Nothing else pins a full run's numbers. Every other test asserts a property, an
invariant, or one component's behaviour, so a refactor that shifts every P&L by a
cent passes all of them. This one stores the actual numbers and compares exactly.

The lake is generated deterministically from a fixed spec, the seed is fixed, and
the money figures are compared in MICRODOLLARS as integers -- an exact equality,
because the engine's central claim is that its accounting is exact and a tolerance
here would quietly concede it.

## When this test fails

It means a number changed. That is not automatically a bug and not automatically a
regression, but it IS always something to explain before proceeding:

  - If the change is intended, re-record with `--record-golden` and put the reason
    and the before/after in the commit message. A re-recording with no explanation
    is how a golden test stops being evidence.
  - If it is not intended, the diff below names which figure moved.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import obt_engine as E
import pytest
from optionsbacktester import runner
from optionsbacktester.report import build_performance_report
from optionsbacktester.strategy import Strategy, buy, group, sell

from tests import fixtures as F

GOLDEN = Path(__file__).parent / "golden" / "replay.json"
TICKER = "TEST"
SEED = 20240102
PATHS = 8


def lake_spec() -> F.LakeSpec:
    """
    Fixed and deliberately varied: a tape that rises, falls, and recovers, so the
    run produces winners, losers, and a drawdown rather than one monotonic outcome.
    """
    return F.LakeSpec(
        ticker=TICKER,
        # Long enough to run past the 30-day expiry, so the replay covers
        # settlement rather than stopping just short of it.
        trading_days=45,
        bars_per_day=5,
        underlying_path=lambda i: 100.0 + 8.0 * math.sin(i / 3.0) + 0.15 * i,
    )


class PoorMansCoveredCall(Strategy):
    """
    Buys one LEAP call and writes short calls against it repeatedly, re-writing
    after each expires or is closed. The structure BACKTESTER_GOALS.md names
    explicitly, and the one that exercises the most engine surface: a long-dated
    long, a rolling short, margin pairing across expiries, and settlement.
    """

    name = "pmcc"

    def __init__(self):
        self.long_leg: int | None = None
        self.short_leg: int | None = None
        self.days_held = 0

    # Sessions after which the short is bought back rather than left to expire, so
    # the replay covers a voluntary close as well as a settlement.
    BUY_BACK_ON = frozenset({12, 30})

    @property
    def parameters(self) -> dict:
        return {"long_target_delta": 0.70, "short_target_delta": 0.30,
                "buy_back_sessions": sorted(self.BUY_BACK_ON)}

    def on_session_start(self, context) -> None:
        self.days_held += 1

    def on_market_snapshot(self, chain, context):
        held = {p.contract_version_id: p.quantity for p in context.positions}

        if self.long_leg is None or self.long_leg not in held:
            leap = chain.calls().expiring_in(300, 500).nearest_delta(0.70)
            if leap is None:
                return ()
            self.long_leg = leap.contract_version_id
            return [group(buy(self.long_leg, 1))]

        if self.short_leg is not None:
            if self.short_leg not in held:
                # Expired or assigned away. Write the next one.
                self.short_leg = None
            elif self.days_held in self.BUY_BACK_ON:
                leg, self.short_leg = self.short_leg, None
                return [group(buy(leg, 1, reduce_only=True))]
            else:
                return ()

        short = chain.calls().expiring_in(5, 40).nearest_delta(0.30)
        if short is None:
            return ()
        self.short_leg = short.contract_version_id
        return [group(sell(self.short_leg, 1))]


def config() -> E.BacktestConfig:
    cfg = E.BacktestConfig()
    cfg.initial_cash = 50_000.0
    cfg.spread_mc_paths = PATHS
    cfg.spread_mc_seed = SEED
    cfg.spread_model.kind = E.SpreadModelKind.CONDITIONAL_LOGNORMAL
    cfg.margin_model = E.MarginModel.ROBINHOOD
    cfg.assignment_policy = E.AssignmentPolicy.CONSERVATIVE_EARLY_ASSIGNMENT
    return cfg


def observed(tmp_path: Path) -> dict:
    """Run the replay and reduce it to the figures worth pinning."""
    F.write_lake(tmp_path, lake_spec())
    result = runner.run(PoorMansCoveredCall, data_root=tmp_path, ticker=TICKER,
                        config=config(), hash_data=True)
    report = build_performance_report(result)
    trades = result.trades[report.path_index]

    return {
        # config_sha256 and engine_sha256 are deliberately absent. Both depend on
        # the compiled extension, and this toolchain does not produce a byte-identical
        # binary from identical source -- two consecutive builds differ. Pinning them
        # here would make every rebuild look like a behaviour change and train a
        # reader to re-record without reading the diff, which is exactly how a golden
        # test stops being evidence. That the config hash covers everything it should
        # is asserted in test_run_manifest.py, where it belongs.
        "manifest": {
            "day_count": result.manifest.day_count,
            "bar_count": result.manifest.bar_count,
            "option_row_count": result.manifest.option_row_count,
            "data_sha256": result.manifest.data_sha256,
        },
        # Every path, in order, so a change to one path's draw is visible rather
        # than averaged away.
        "net_pnl_micros": [p.net_pnl_micros for p in result.paths],
        "fill_counts": [p.fill_count for p in result.paths],
        "rejection_counts": [p.rejection_count for p in result.paths],
        "trade_counts": [p.trade_count for p in result.paths],
        "spread_cost_micros": [p.spread_cost_micros for p in result.paths],
        "fees_micros": [p.fees_micros for p in result.paths],
        "max_drawdown_micros": [round(p.max_drawdown * 1_000_000) for p in result.paths],
        "peak_margin_micros": [round(p.peak_margin_requirement * 1_000_000)
                               for p in result.paths],
        "assignment_count": [p.assignment_count for p in result.paths],
        "early_assignment_count": [p.early_assignment_count for p in result.paths],
        "exercise_count": [p.exercise_count for p in result.paths],
        "expiration_count": [p.expiration_count for p in result.paths],
        "quarantined_positions": [p.quarantined_positions for p in result.paths],
        "ledger_reconciles": [p.ledger_reconciles for p in result.paths],
        "deterministic_pnl_micros": round(result.deterministic_pnl * 1_000_000),
        "representative_path": report.path_index,
        # The median path's round trips, exactly.
        "trades": [
            {
                "cv": t.contract_version_id,
                "qty": t.quantity,
                "short": t.was_short,
                "reason": str(t.reason).rsplit(".", 1)[-1],
                "pnl_micros": t.realized_pnl_micros,
                "fees_micros": t.fees_micros,
                "spread_micros": t.spread_cost_micros,
            }
            for t in trades
        ],
    }


@pytest.fixture(scope="module")
def replay(tmp_path_factory) -> dict:
    return observed(tmp_path_factory.mktemp("golden"))


def test_the_golden_file_exists(replay):
    """
    Fails with instructions rather than a KeyError if the file is missing. Record
    with: uv run python tests/record_golden.py
    """
    assert GOLDEN.exists(), (
        f"{GOLDEN} is missing. Record it with:\n"
        "    uv run python tests/record_golden.py\n"
        "and commit it with the reason in the message."
    )


@pytest.fixture(scope="module")
def expected() -> dict:
    if not GOLDEN.exists():
        pytest.skip("golden file not recorded")
    return json.loads(GOLDEN.read_text())


class TestTheReplayIsReproducible:
    def test_the_same_run_twice_agrees_exactly(self, tmp_path_factory):
        """
        Before comparing against a stored file, the run has to be deterministic at
        all. Two invocations over separately generated lakes must agree.
        """
        first = observed(tmp_path_factory.mktemp("a"))
        second = observed(tmp_path_factory.mktemp("b"))

        assert first == second

    def test_the_lake_is_byte_identical_across_generations(self, tmp_path_factory):
        one = tmp_path_factory.mktemp("l1")
        two = tmp_path_factory.mktemp("l2")
        F.write_lake(one, lake_spec())
        F.write_lake(two, lake_spec())

        for a in sorted(one.rglob("*.parquet")):
            b = two / a.relative_to(one)
            assert a.read_bytes() == b.read_bytes(), a.name


class TestTheReplayMatchesTheGoldenFile:
    def test_the_manifest_matches(self, replay, expected):
        assert replay["manifest"] == expected["manifest"]

    @pytest.mark.parametrize("key", [
        "net_pnl_micros", "fill_counts", "rejection_counts", "trade_counts",
        "spread_cost_micros", "fees_micros", "max_drawdown_micros",
        "peak_margin_micros", "assignment_count", "early_assignment_count",
        "exercise_count", "expiration_count", "quarantined_positions",
        "ledger_reconciles",
    ])
    def test_a_per_path_figure_matches(self, replay, expected, key):
        assert replay[key] == expected[key], (
            f"{key} changed.\n  was {expected[key]}\n  now {replay[key]}"
        )

    def test_the_deterministic_figure_matches(self, replay, expected):
        assert replay["deterministic_pnl_micros"] == expected["deterministic_pnl_micros"]

    def test_every_round_trip_matches(self, replay, expected):
        assert len(replay["trades"]) == len(expected["trades"])
        for i, (now, then) in enumerate(zip(replay["trades"], expected["trades"],
                                            strict=True)):
            assert now == then, f"trade {i} changed:\n  was {then}\n  now {now}"


class TestTheReplayIsWorthPinning:
    """
    A golden file over a run that did nothing would pass forever and prove nothing.
    """

    def test_the_run_actually_traded(self, replay):
        assert min(replay["fill_counts"]) > 10
        assert min(replay["trade_counts"]) > 3

    def test_the_run_covered_both_a_voluntary_close_and_a_settlement(self):
        """
        A replay that only ever closed positions voluntarily would not pin the
        settlement path at all.
        """
        import json
        recorded = json.loads(GOLDEN.read_text())
        reasons = {t["reason"] for t in recorded["trades"]}

        assert "CLOSED" in reasons
        assert reasons & {"ASSIGNED", "EXPIRED", "EXERCISED"}

    def test_the_run_produced_both_winners_and_losers(self, replay):
        pnls = [t["pnl_micros"] for t in replay["trades"]]

        assert any(p > 0 for p in pnls)
        assert any(p < 0 for p in pnls)

    def test_the_run_exercised_settlement(self, replay):
        assert sum(replay["expiration_count"]) > 0

    def test_the_run_had_a_drawdown(self, replay):
        assert min(replay["max_drawdown_micros"]) > 0

    def test_the_paths_actually_differ(self, replay):
        """
        Identical paths would mean the Monte Carlo contributed nothing, and the
        golden file would not be pinning the spread model at all.
        """
        assert len(set(replay["net_pnl_micros"])) > 1

    def test_every_path_reconciles(self, replay):
        assert all(replay["ledger_reconciles"])

    def test_the_short_leg_was_rewritten_more_than_once(self, replay):
        """
        The poor man's covered call only exercises what it is meant to if the short
        call is written repeatedly against one long.
        """
        shorts = [t for t in replay["trades"] if t["short"]]

        assert len(shorts) > 1
