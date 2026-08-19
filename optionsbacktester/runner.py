"""
Backtest runner.

All Monte Carlo paths advance in lockstep over a single pass of the data. The
alternative -- replaying the lake once per path -- would re-read and re-decode
every Parquet file N times for identical market data, and 1,000 paths would mean
1,000 passes. Here the data is read once and handed to N engines, so cost scales
with N only in portfolio state, which is small.

That also makes common random numbers automatic: every path sees the same bars
at the same instants, so a difference between paths can only come from the
spread draw, which is the only stochastic component.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

import obt_engine as E

from .contracts import (
    build_lineage_transitions,
    build_snapshot,
    contract_version_key,
    to_ns,
)
from .contracts import build_contracts
from .stream import DataLake, DaySlice, UniverseFilter, iter_days, iter_timestamp_batches
from .strategy import Chain, Context, Strategy, chain_from_batch


def _row_key(row: dict) -> int:
    return contract_version_key(
        row["symbol"],
        float(row.get("strike") or 0.0),
        float(row.get("deliverable_equity_amount") or 100.0),
        float(row.get("quote_multiplier") or 100.0),
    )


@dataclass
class RunManifest:
    """
    Everything needed to reproduce a run.

    Recorded because a result nobody can reproduce is not evidence. The data
    hash covers the actual files read, so silently swapping a day's Parquet
    changes the manifest.
    """
    ticker: str
    start: str
    end: str
    strategy: str
    engine_version: str
    spread_mc_paths: int
    spread_mc_seed: int
    spread_model: str
    execution_timing: str
    assignment_policy: str
    margin_model: str
    fee_schedule: str
    initial_cash: float
    data_sha256: str = ""
    day_count: int = 0
    config_sha256: str = ""
    platform: str = field(default_factory=lambda: f"{platform.system()}-{platform.machine()}")

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def finalize(self) -> RunManifest:
        payload = json.dumps(
            {k: v for k, v in self.to_dict().items() if k != "config_sha256"},
            sort_keys=True,
        )
        self.config_sha256 = hashlib.sha256(payload.encode()).hexdigest()
        return self


@dataclass
class RunResult:
    manifest: RunManifest
    paths: list[E.PathMetrics]
    equity_curves: list[list[float]]
    fills: list[E.Fill]
    rejections: list[E.OrderRejection]
    # Per-path trade ledgers and decomposed equity series. Both are needed for
    # trade-level statistics: aggregating over paths alone cannot say anything
    # about the distribution of individual trades.
    trades: list[list[E.TradeRecord]] = field(default_factory=list)
    equity_points: list[list[E.EquityPoint]] = field(default_factory=list)

    @property
    def deterministic(self) -> bool:
        """True when every path produced the same P&L, i.e. spread cost was zero."""
        if not self.paths:
            return True
        first = self.paths[0].net_pnl_micros
        return all(p.net_pnl_micros == first for p in self.paths)

    @property
    def representative(self) -> int:
        """
        Index of the path whose net P&L is the median.

        Trade-level and equity-curve reporting has to pick a path, and the median
        is the defensible choice: the mean path does not exist, and the first path
        is an arbitrary draw.
        """
        if not self.paths:
            return 0
        order = sorted(range(len(self.paths)), key=lambda i: self.paths[i].net_pnl_micros)
        return order[len(order) // 2]


def _hash_files(paths: list[Path]) -> str:
    """Content hash over the files actually read, in a stable order."""
    digest = hashlib.sha256()
    for p in sorted(paths):
        digest.update(p.name.encode())
        try:
            digest.update(str(p.stat().st_size).encode())
            with open(p, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    digest.update(chunk)
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()


def _spread_model_name(cfg: E.SpreadModelConfig) -> str:
    return str(cfg.kind).rsplit(".", 1)[-1].lower()


def run(
    strategy_factory,
    *,
    data_root: str | Path,
    ticker: str,
    config: E.BacktestConfig,
    start: date | None = None,
    end: date | None = None,
    universe: UniverseFilter | None = None,
    require_complete_days: bool = True,
    hash_data: bool = True,
) -> RunResult:
    """
    Run a strategy across all Monte Carlo paths.

    ``strategy_factory`` is called once per path so each path gets its own
    strategy instance and cannot leak state into another.
    """
    lake = DataLake(data_root, ticker)
    path_count = max(1, int(config.spread_mc_paths))

    engines: list[E.Engine] = []
    strategies: list[Strategy] = []
    for scenario in range(path_count):
        engine = E.Engine(config)
        engine.begin_scenario(scenario)
        engines.append(engine)
        strategies.append(strategy_factory())

    contracts: dict[int, E.OptionContractVersion] = {}
    files_read: list[Path] = []
    day_count = 0
    session_started = [False] * path_count

    for day in iter_days(lake, start, end, universe, require_complete_days):
        day_count += 1
        if hash_data:
            day_dir = lake.ticker_dir / f"{day.day.year:04d}" / f"{day.day.month:02d}" / f"{day.day.day:02d}"
            files_read.extend(sorted(day_dir.glob("*.parquet")))

        new_contracts = build_contracts(day.options, ticker)
        contracts.update(new_contracts)
        for engine in engines:
            engine.set_contracts(list(contracts.values()))

        transitions = build_lineage_transitions(day.lineage_events, contracts)
        if transitions:
            for engine in engines:
                engine.queue_corporate_actions(transitions)

        _run_day(day, engines, strategies, contracts, ticker, session_started)

    results = [engine.finalize() for engine in engines]
    manifest = RunManifest(
        ticker=ticker,
        start=str(start) if start else "",
        end=str(end) if end else "",
        strategy=getattr(strategies[0], "name", strategies[0].__class__.__name__),
        engine_version="obt-engine-1",
        spread_mc_paths=path_count,
        spread_mc_seed=int(config.spread_mc_seed),
        spread_model=_spread_model_name(config.spread_model),
        execution_timing=str(config.execution_timing).rsplit(".", 1)[-1].lower(),
        assignment_policy=E.assignment_policy_name(config.assignment_policy),
        margin_model=str(config.margin_model).rsplit(".", 1)[-1].lower(),
        fee_schedule=config.fees.schedule_id,
        initial_cash=config.initial_cash,
        data_sha256=_hash_files(files_read) if hash_data else "",
        day_count=day_count,
    ).finalize()

    return RunResult(
        manifest=manifest,
        paths=results,
        equity_curves=[e.equity_curve() for e in engines],
        fills=list(engines[0].fills()),
        rejections=list(engines[0].rejections()),
        trades=[list(e.trades()) for e in engines],
        equity_points=[list(e.equity_points()) for e in engines],
    )


def _run_day(
    day: DaySlice,
    engines: list[E.Engine],
    strategies: list[Strategy],
    contracts: dict[int, E.OptionContractVersion],
    ticker: str,
    session_started: list[bool],
) -> None:
    """
    One trading day, following the spec's required ordering per timestamp.

    Steps 1-3 and the deferred fill happen in begin_bar; the strategy sees only
    what step 4 exposes; step 5 submits; steps 7-9 happen in end_bar. The session
    is then closed explicitly, which is what settles expirations: no bar occupies
    the 16:00 expiration instant, so settling on bar timestamps alone would defer
    every expiry to the next session's open.
    """
    session_day = datetime.combine(day.day, datetime.min.time())
    # Anything expiring at any hour of this calendar day settles at its close.
    session_close_ns = to_ns(session_day + timedelta(days=1))
    delivered_fills = [len(e.fills()) for e in engines]

    # on_session_start must pair with on_session_end once per session. It
    # previously fired once for the entire run while on_session_end fired daily,
    # so any per-session state a strategy kept was never reset.
    for i, strategy in enumerate(strategies):
        strategy.on_session_start(Context(
            timestamp=session_day,
            account=engines[i].account_state(),
            positions=engines[i].positions(),
            equity_positions=engines[i].equity_positions(),
            contracts=contracts,
            session_day=session_day,
            scenario_id=i,
        ))
        session_started[i] = True

    for ts, batch in iter_timestamp_batches(day):
        snapshot = build_snapshot(ts, batch, contracts, ticker)
        chain = chain_from_batch(batch, contracts, _row_key)

        for i, (engine, strategy) in enumerate(zip(engines, strategies, strict=True)):
            engine.begin_bar(snapshot)

            context = Context(
                timestamp=ts,
                account=engine.account_state(),
                positions=engine.positions(),
                equity_positions=engine.equity_positions(),
                contracts=contracts,
                session_day=session_day,
                scenario_id=i,
            )

            for order_group in strategy.on_market_snapshot(chain, context) or ():
                engine.submit_group(order_group)

            engine.end_bar()

            # on_fill and on_corporate_action were declared in the Strategy API
            # and never invoked, so any strategy logic in them silently never ran.
            for fill in engine.fills()[delivered_fills[i]:]:
                strategy.on_fill(fill, context)
            delivered_fills[i] = len(engine.fills())

    for i, (engine, strategy) in enumerate(zip(engines, strategies, strict=True)):
        engine.end_session(session_close_ns)
        strategy.on_session_end(Context(
            timestamp=session_day,
            account=engine.account_state(),
            positions=engine.positions(),
            equity_positions=engine.equity_positions(),
            contracts=contracts,
            session_day=session_day,
            scenario_id=i,
        ))
