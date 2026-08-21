"""
The run manifest.

A result nobody can reproduce is not evidence. The config hash previously covered
only the manifest's own summary fields, so two runs with different spread
calibrations, different risk limits, or different fail-closed gates reported the
SAME config_sha256 and produced different numbers -- which is worse than having no
hash, because it actively asserts equivalence.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import obt_engine as E
import pytest
from optionsbacktester import runner
from optionsbacktester.strategy import Strategy, buy, group, sell
from optionsbacktester.stream import UniverseFilter

from tests import fixtures as F

TICKER = "TEST"


class Churn(Strategy):
    name = "churn"

    def __init__(self, target_delta: float = 0.40):
        self.target_delta = target_delta
        self.held = None

    @property
    def parameters(self) -> dict:
        return {"target_delta": self.target_delta}

    def on_market_snapshot(self, chain, context):
        if self.held is not None:
            out = [group(sell(self.held, 1, reduce_only=True))]
            self.held = None
            return out
        pick = chain.calls().expiring_in(10, 70).nearest_delta(self.target_delta)
        if pick is None:
            return ()
        self.held = pick.contract_version_id
        return [group(buy(self.held, 1))]


class Anonymous(Strategy):
    """Declares no parameters, which the manifest has to say rather than assume."""

    name = "anonymous"

    def on_market_snapshot(self, chain, context):
        return ()


@pytest.fixture(scope="module")
def lake() -> Path:
    root = Path(tempfile.mkdtemp())
    F.write_lake(root, F.LakeSpec(
        trading_days=6, bars_per_day=4,
        underlying_path=lambda i: 100.0 + 2.0 * math.sin(i),
    ))
    return root


def base_config(**overrides) -> E.BacktestConfig:
    cfg = E.BacktestConfig()
    cfg.initial_cash = 100_000.0
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def run(lake, *, factory=Churn, config=None, universe=None):
    return runner.run(factory, data_root=lake, ticker=TICKER,
                      config=config or base_config(), universe=universe)


def hash_of(lake, **kw) -> str:
    return run(lake, **kw).manifest.config_sha256


class TestTheHashIsStable:
    def test_two_identical_runs_agree(self, lake):
        assert hash_of(lake) == hash_of(lake)

    def test_the_detail_is_kept_alongside_the_hash(self, lake):
        """A mismatch should be diagnosable, not only detectable."""
        detail = run(lake).manifest.config_detail

        assert set(detail) == {"engine_config", "universe", "strategy",
                               "require_complete_days"}
        assert detail["engine_config"]["spread_mc_seed"] == 42


class TestEveryThingThatChangesANumberChangesTheHash:
    """
    Each of these produced an identical config_sha256 before the fingerprint
    covered the whole config.
    """

    def _differs(self, lake, **overrides) -> bool:
        return hash_of(lake) != hash_of(lake, config=base_config(**overrides))

    def test_a_different_seed(self, lake):
        assert self._differs(lake, spread_mc_seed=99)

    def test_a_different_path_count(self, lake):
        assert self._differs(lake, spread_mc_paths=4)

    @pytest.mark.parametrize("field,value", [
        ("require_occ_confirmed_lineage", False),
        ("reject_fallback_analytics", False),
        ("reject_stale_bars", False),
        ("require_point_in_time_terms", False),
        ("require_monotonic_time", False),
        ("mark_age_limit_ns", 0),
    ])
    def test_a_fail_closed_gate(self, lake, field, value):
        """Turning a gate off changes which orders fill. It has to change the hash."""
        assert self._differs(lake, **{field: value})

    def test_a_spread_calibration_parameter(self, lake):
        cfg = base_config()
        cfg.spread_model.median_full_spread_bps = 90.0

        assert hash_of(lake) != hash_of(lake, config=cfg)

    def test_the_spread_variance_scale(self, lake):
        """The knob the report tells a reader to vary. It must be recorded."""
        cfg = base_config()
        cfg.spread_model.variance_scale = 2.0

        assert hash_of(lake) != hash_of(lake, config=cfg)

    def test_a_conditional_beta(self, lake):
        cfg = base_config()
        cfg.spread_model.beta_iv = 1.5

        assert hash_of(lake) != hash_of(lake, config=cfg)

    def test_an_equity_spread_parameter(self, lake):
        cfg = base_config()
        cfg.spread_model.equity_full_spread_bps = 5.0

        assert hash_of(lake) != hash_of(lake, config=cfg)

    def test_a_fee_rate_without_a_new_schedule_id(self, lake):
        """
        A schedule_id is a label. Changing a rate while keeping the label must still
        change the hash, and the old hash covered only the label.
        """
        cfg = base_config()
        cfg.fees.regulatory_per_contract = 0.10
        assert cfg.fees.schedule_id == E.BacktestConfig().fees.schedule_id

        assert hash_of(lake) != hash_of(lake, config=cfg)

    def test_a_risk_limit(self, lake):
        cfg = base_config()
        cfg.risk.max_open_positions = 3

        assert hash_of(lake) != hash_of(lake, config=cfg)

    def test_a_universe_filter(self, lake):
        filtered = run(lake, universe=UniverseFilter(min_dte=5)).manifest

        assert filtered.config_sha256 != hash_of(lake)
        assert filtered.config_detail["universe"]["filtered"] is True

    def test_a_strategy_parameter(self, lake):
        """
        Same strategy class, different target delta: a different experiment that
        nothing in the engine config distinguishes.
        """
        wide = runner.run(lambda: Churn(0.20), data_root=lake, ticker=TICKER,
                          config=base_config())

        assert wide.manifest.config_sha256 != hash_of(lake)
        assert wide.manifest.config_detail["strategy"]["values"]["target_delta"] == 0.20


class TestUnhashableStrategyStateIsDeclared:
    def test_a_strategy_with_no_parameters_says_so(self, lake):
        detail = run(lake, factory=Anonymous).manifest.config_detail["strategy"]

        assert detail["available"] is False
        assert "declares no" in detail["reason"]

    def test_probing_for_parameters_does_not_construct_an_extra_strategy(self, lake):
        """
        Reading them from an instance the run already built, not from a probe. A
        factory that counts its calls would otherwise see one too many.
        """
        built: list[Churn] = []

        def factory():
            s = Churn()
            built.append(s)
            return s

        cfg = base_config()
        cfg.spread_mc_paths = 2
        runner.run(factory, data_root=lake, ticker=TICKER, config=cfg)

        # Two paths plus the zero-spread reference engine, and nothing more.
        assert len(built) == 3


class TestEngineBinaryIdentity:
    def test_the_manifest_records_the_engine_hash(self, lake):
        """
        The engine's behaviour lives in the compiled extension, so a rebuild is a
        different experiment even at an identical config, and nothing else in the
        manifest would show it.
        """
        engine_hash = run(lake).manifest.engine_sha256

        assert len(engine_hash) == 64
        assert engine_hash == hashlib_of(E.__file__)


def hashlib_of(path: str) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
