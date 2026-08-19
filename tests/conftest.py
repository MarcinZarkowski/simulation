"""Shared fixtures and builders for engine tests."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import obt_engine as E  # noqa: E402

DAY_NS = 86_400 * 10**9
EPOCH = datetime(1970, 1, 1)


def ns(value: datetime) -> int:
    delta = value - EPOCH
    return (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000


def day_ns(day: int) -> int:
    """Nanoseconds for a whole number of days after the epoch."""
    return day * DAY_NS


def make_contract(
    cv_id: int,
    *,
    symbol: str | None = None,
    strike: float = 100.0,
    expiry_day: int = 30,
    is_call: bool = True,
    multiplier: int = 100,
    deliverable_shares: float | None = None,
    deliverable_cash: float = 0.0,
    underlying: str = "TEST",
    tradable: bool = True,
    analytics: bool = True,
    is_adjusted: bool = False,
    instrument_id: int | None = None,
) -> E.OptionContractVersion:
    c = E.OptionContractVersion()
    c.id = cv_id
    c.instrument_id = instrument_id if instrument_id is not None else cv_id
    c.symbol = symbol or f"{underlying}{expiry_day:06d}{'C' if is_call else 'P'}{int(strike * 1000):08d}"
    c.underlying_symbol = underlying
    c.type = E.OptionType.CALL if is_call else E.OptionType.PUT
    c.strike = strike
    c.pricing_strike = strike
    c.quote_multiplier = multiplier
    shares = multiplier if deliverable_shares is None else deliverable_shares
    c.deliverable_equity_microshares = int(round(shares * 1_000_000))
    c.deliverable_cash = deliverable_cash
    c.expiration = day_ns(expiry_day)
    c.valid_from = 0
    c.valid_to = c.expiration
    c.tradable_for_new_positions = tradable
    c.analytics_supported = analytics
    c.is_adjusted = is_adjusted
    return c


def make_bar(
    cv_id: int,
    *,
    timestamp_ns: int,
    price: float,
    open_price: float | None = None,
    volume: int = 500,
    trade_count: int = 50,
    stale: bool = False,
    analytics_valid: bool = True,
) -> E.MarketBar:
    b = E.MarketBar()
    b.timestamp = timestamp_ns
    b.contract_version_id = cv_id
    b.open = price if open_price is None else open_price
    b.high = price
    b.low = price
    b.close = price
    b.vwap = price
    b.valuation_price = price
    b.volume = volume
    b.trade_count = trade_count
    b.stale = stale
    b.analytics_valid = analytics_valid
    return b


def make_analytics(cv_id: int, *, timestamp_ns: int, delta: float = 50.0,
                   iv: float = 0.25, valid: bool = True) -> E.OptionAnalytics:
    a = E.OptionAnalytics()
    a.timestamp = timestamp_ns
    a.contract_version_id = cv_id
    a.implied_volatility = iv
    a.delta = delta
    a.valid = valid
    return a


def make_snapshot(timestamp_ns: int, bars: list[E.MarketBar], *,
                  underlying: dict[str, float] | None = None,
                  analytics: list[E.OptionAnalytics] | None = None) -> E.MarketSnapshot:
    s = E.MarketSnapshot()
    s.timestamp = timestamp_ns
    s.bars = bars
    s.analytics = analytics or []
    s.underlying_price = underlying or {"TEST": 100.0}
    return s


def base_config(
    *,
    cash: float = 100_000.0,
    paths: int = 1,
    seed: int = 42,
    spread: E.SpreadModelKind = E.SpreadModelKind.ZERO,
    fees: bool = False,
    margin: E.MarginModel = E.MarginModel.REG_T,
    timing: E.ExecutionTiming = E.ExecutionTiming.NEXT_BAR_OPEN,
    assignment: E.AssignmentPolicy = E.AssignmentPolicy.AUTOMATIC_ITM_EXERCISE,
    reject_stale: bool = True,
    reject_fallback: bool = True,
) -> E.BacktestConfig:
    """
    Config with every stochastic and frictional component off by default.

    Tests that assert an exact ledger value need zero spread and zero fees;
    tests about spread or fees turn exactly one of them on.
    """
    cfg = E.BacktestConfig()
    cfg.initial_cash = cash
    cfg.spread_mc_paths = paths
    cfg.spread_mc_seed = seed
    cfg.spread_model.kind = spread
    cfg.margin_model = margin
    cfg.execution_timing = timing
    cfg.assignment_policy = assignment
    cfg.reject_stale_bars = reject_stale
    cfg.reject_fallback_analytics = reject_fallback
    if not fees:
        cfg.fees = E.FeeSchedule.zero()
    return cfg


class EngineHarness:
    """
    Drives an engine bar by bar without the streaming layer.

    Keeps the ordering explicit: begin_bar, submit, end_bar. Orders submitted on
    one bar fill on the next, which is what the default timing requires.
    """

    def __init__(self, config: E.BacktestConfig, contracts: list[E.OptionContractVersion]):
        self.engine = E.Engine(config)
        self.engine.set_contracts(contracts)
        self.engine.begin_scenario(0)
        self.contracts = {c.id: c for c in contracts}

    def bar(self, timestamp_ns: int, bars: list[E.MarketBar], *,
            groups: list[E.OrderGroup] | None = None,
            underlying: dict[str, float] | None = None,
            analytics: list[E.OptionAnalytics] | None = None):
        self.engine.begin_bar(make_snapshot(timestamp_ns, bars, underlying=underlying,
                                            analytics=analytics))
        for g in groups or []:
            self.engine.submit_group(g)
        self.engine.end_bar()
        return self.engine.account_state()

    def finalize(self) -> E.PathMetrics:
        return self.engine.finalize()

    @property
    def cash(self) -> float:
        return self.engine.cash()

    @property
    def cash_micros(self) -> int:
        return self.engine.cash_micros()

    def positions(self):
        return self.engine.positions()

    def equity_positions(self):
        return self.engine.equity_positions()

    def fills(self):
        return self.engine.fills()

    def rejections(self):
        return self.engine.rejections()

    def quantity_of(self, cv_id: int) -> int:
        for p in self.positions():
            if p.contract_version_id == cv_id:
                return p.quantity
        return 0

    def shares_of(self, symbol: str) -> int:
        for e in self.equity_positions():
            if e.symbol == symbol:
                return e.shares
        return 0


@pytest.fixture
def tmp_lake(tmp_path):
    """Factory for a synthetic pipeline-format lake."""
    from tests import fixtures as F

    def build(kind: str = "flat", **kwargs):
        if kind == "ramp":
            return F.ramp_lake(tmp_path, **kwargs)
        if kind == "crash":
            return F.crash_lake(tmp_path, **kwargs)
        return F.flat_lake(tmp_path, **kwargs)

    return build
