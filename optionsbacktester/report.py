"""
Monte Carlo aggregation.

Deterministic market-data P&L and stochastic execution cost are reported
separately. Conflating them would let a spread assumption masquerade as a
result: without calibrated quote history the spread distribution is a
sensitivity analysis, and the report says so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import obt_engine as E

from .runner import RunResult


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile on an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


@dataclass
class MonteCarloReport:
    paths: int
    seed: int
    spread_model: str

    mean_net_pnl: float
    median_net_pnl: float
    stdev_net_pnl: float
    standard_error: float
    ci_low: float
    ci_high: float
    confidence_level: float

    worst_net_pnl: float
    best_net_pnl: float
    probability_of_profit: float
    margin_breach_probability: float

    mean_spread_cost: float
    mean_fees: float
    mean_max_drawdown: float
    mean_return_fraction: float

    # P&L with execution cost removed. Identical across paths by construction,
    # so it isolates what the market data alone implies.
    deterministic_net_pnl: float

    percentiles: dict[str, float]
    total_fills: int
    total_rejections: int
    ledger_reconciles: bool

    def summary(self) -> str:
        pct = self.confidence_level * 100
        lines = [
            f"Monte Carlo over {self.paths} path(s), seed {self.seed}, model {self.spread_model}",
            "",
            f"  net P&L mean        {self.mean_net_pnl:>14,.2f}",
            f"  net P&L median      {self.median_net_pnl:>14,.2f}",
            f"  net P&L stdev       {self.stdev_net_pnl:>14,.2f}",
            f"  MC standard error   {self.standard_error:>14,.2f}",
            f"  {pct:.0f}% interval      [{self.ci_low:,.2f}, {self.ci_high:,.2f}]",
            f"  worst / best        {self.worst_net_pnl:>14,.2f} / {self.best_net_pnl:,.2f}",
            f"  P(profit)           {self.probability_of_profit:>14.1%}",
            f"  P(margin breach)    {self.margin_breach_probability:>14.1%}",
            "",
            f"  before spread cost  {self.deterministic_net_pnl:>14,.2f}   (deterministic)",
            f"  mean spread cost    {self.mean_spread_cost:>14,.2f}   (stochastic)",
            f"  mean fees           {self.mean_fees:>14,.2f}",
            f"  mean max drawdown   {self.mean_max_drawdown:>14,.2f}",
            "",
            f"  fills {self.total_fills}   rejections {self.total_rejections}"
            f"   ledger reconciles {self.ledger_reconciles}",
        ]
        return "\n".join(lines)


def build_report(result: RunResult, confidence_level: float = 0.95) -> MonteCarloReport:
    paths = result.paths
    if not paths:
        raise ValueError("no scenarios were run")

    net = sorted(p.net_pnl for p in paths)
    n = len(net)
    mean = sum(net) / n
    variance = sum((x - mean) ** 2 for x in net) / (n - 1) if n > 1 else 0.0
    stdev = math.sqrt(variance)
    stderr = stdev / math.sqrt(n) if n > 0 else 0.0

    tail = (1.0 - confidence_level) / 2.0
    spread_costs = [p.spread_cost for p in paths]
    mean_spread = sum(spread_costs) / n

    return MonteCarloReport(
        paths=n,
        seed=result.manifest.spread_mc_seed,
        spread_model=result.manifest.spread_model,
        mean_net_pnl=mean,
        median_net_pnl=_percentile(net, 0.5),
        stdev_net_pnl=stdev,
        standard_error=stderr,
        ci_low=_percentile(net, tail),
        ci_high=_percentile(net, 1.0 - tail),
        confidence_level=confidence_level,
        worst_net_pnl=net[0],
        best_net_pnl=net[-1],
        probability_of_profit=sum(1 for x in net if x > 0) / n,
        margin_breach_probability=sum(1 for p in paths if p.margin_breached) / n,
        mean_spread_cost=mean_spread,
        mean_fees=sum(p.fees for p in paths) / n,
        mean_max_drawdown=sum(p.max_drawdown for p in paths) / n,
        mean_return_fraction=sum(p.return_fraction for p in paths) / n,
        # Adding execution cost back recovers the no-spread result.
        deterministic_net_pnl=mean + mean_spread,
        percentiles={
            "p05": _percentile(net, 0.05),
            "p25": _percentile(net, 0.25),
            "p50": _percentile(net, 0.50),
            "p75": _percentile(net, 0.75),
            "p95": _percentile(net, 0.95),
        },
        total_fills=sum(p.fill_count for p in paths),
        total_rejections=sum(p.rejection_count for p in paths),
        ledger_reconciles=all(p.ledger_reconciles for p in paths),
    )


def convergence_table(results: dict[int, RunResult]) -> str:
    """
    Standard error against path count.

    Monte Carlo error falls as 1/sqrt(n), so quadrupling paths should roughly
    halve the standard error. A table that does not is evidence the draws are
    correlated rather than independent.
    """
    lines = [f"{'paths':>8}{'mean P&L':>16}{'stdev':>14}{'std error':>14}"]
    for paths in sorted(results):
        r = build_report(results[paths])
        lines.append(
            f"{r.paths:>8}{r.mean_net_pnl:>16,.2f}{r.stdev_net_pnl:>14,.2f}{r.standard_error:>14,.2f}"
        )
    return "\n".join(lines)
