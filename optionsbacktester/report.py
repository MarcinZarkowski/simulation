"""
Reporting.

Two things are kept apart throughout, because conflating them is how a reader
mistakes an assumption for a measurement:

  - **Across paths**: the Monte Carlo distribution of the result. This describes
    uncertainty in execution cost and nothing else.
  - **Within a path**: the trades and the account curve. This describes what the
    strategy did.

Deterministic market-data P&L is reported separately from stochastic spread cost,
and the deterministic figure comes from an actual zero-spread run rather than from
adding mean spread cost back to the mean.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

import obt_engine as E

from .analytics import (
    AccountStats,
    TradeStats,
    account_stats,
    percentile,
    sparkline,
    trade_stats,
)
from .runner import RunResult


def _fmt(amount: float, width: int = 14) -> str:
    return f"{amount:>{width},.2f}"


@dataclass
class MonteCarloReport:
    """Distribution of the result across Monte Carlo paths."""
    paths: int
    seed: int
    spread_model: str
    variance_scale: float

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

    # From a real zero-spread run.
    deterministic_net_pnl: float
    # deterministic - (mean + mean_spread). Nonzero means the spread changed which
    # orders filled, so the two components are not additively separable.
    separability_residual: float

    percentiles: dict[str, float]
    total_fills: int
    total_rejections: int
    mean_dividend_cash: float
    total_early_assignments: int
    stale_mark_valuations: int
    max_mark_age_days: float
    settlements_without_official_price: int
    ledger_reconciles: bool
    degenerate: bool
    truncated_paths: int
    quarantined_positions: int

    def summary(self) -> str:
        pct = self.confidence_level * 100
        lines = [
            f"Monte Carlo over {self.paths} path(s), seed {self.seed}, "
            f"model {self.spread_model}, variance x{self.variance_scale:g}",
            "",
            f"  net P&L mean        {_fmt(self.mean_net_pnl)}",
            f"  net P&L median      {_fmt(self.median_net_pnl)}",
            f"  net P&L stdev       {_fmt(self.stdev_net_pnl)}",
            f"  MC standard error   {_fmt(self.standard_error)}",
        ]
        if self.degenerate:
            lines += [
                f"  {pct:.0f}% interval      not reported",
                "",
                "  Every path produced the same P&L, so there is no interval to report.",
                "  This happens when the drawn spread is pinned at the minimum tick for",
                "  every fill -- common for cheap options, where a one-cent quote really",
                "  is the whole spread. It is not a precise result; it is an absent one.",
            ]
        else:
            lines += [
                f"  {pct:.0f}% interval      [{self.ci_low:,.2f}, {self.ci_high:,.2f}]",
                f"  worst / best        {_fmt(self.worst_net_pnl)} / {self.best_net_pnl:,.2f}",
                f"  P(profit)           {self.probability_of_profit:>14.1%}",
            ]
        lines += [
            f"  P(margin breach)    {self.margin_breach_probability:>14.1%}",
            "",
            f"  before spread cost  {_fmt(self.deterministic_net_pnl)}   (zero-spread run)",
            f"  mean spread cost    {_fmt(self.mean_spread_cost)}   (stochastic)",
            f"  mean fees           {_fmt(self.mean_fees)}",
            f"  mean max drawdown   {_fmt(self.mean_max_drawdown)}",
        ]
        if self.mean_dividend_cash:
            lines.append(
                f"  mean dividend cash  {_fmt(self.mean_dividend_cash)}"
                "   (negative = owed on short shares)")
        if self.total_early_assignments:
            lines += [
                "",
                f"  early assignments   {self.total_early_assignments:>14}",
                "  Short calls taken before an ex-dividend date, where the dividend",
                "  exceeded the extrinsic value the holder gave up by exercising.",
            ]
        if abs(self.separability_residual) > 0.005:
            lines += [
                "",
                f"  separability residual {self.separability_residual:>12,.2f}",
                "  The spread changed which orders filled, so deterministic P&L and",
                "  spread cost are not additively separable on this run.",
            ]
        lines += [
            "",
            f"  fills {self.total_fills}   rejections {self.total_rejections}"
            f"   ledger reconciles {self.ledger_reconciles}",
        ]
        if self.stale_mark_valuations:
            lines += [
                "",
                f"  stale-mark valuations {self.stale_mark_valuations:>11}"
                f"   oldest mark {self.max_mark_age_days:.1f} day(s)",
                "  Those positions were valued at intrinsic against a fresh underlying,",
                "  because the contract itself had stopped printing. Intrinsic is a floor,",
                "  so an out-of-the-money position marked this way reads as worthless.",
            ]
        elif self.max_mark_age_days >= 1.0:
            lines += [
                "",
                f"  oldest mark used    {self.max_mark_age_days:>14.1f} day(s)",
            ]
        if self.settlements_without_official_price:
            lines += [
                "",
                f"  settled without an official value: "
                f"{self.settlements_without_official_price}",
                "  A cash-settled contract resolved against the last observed spot rather",
                "  than a published settlement value, which is computed from opening prints",
                "  and can differ materially.",
            ]
        if self.truncated_paths:
            lines += [
                "",
                f"  TRUNCATED: {self.truncated_paths} of {self.paths} path(s) quarantined "
                f"{self.quarantined_positions} position(s).",
                "  A contract adjustment could not be sourced, so the affected position was",
                "  closed at its last observed mark rather than carried through a conversion",
                "  the engine cannot justify. P&L past that point describes a reduced book.",
            ]
        return "\n".join(lines)


@dataclass
class PerformanceReport:
    """Everything about one representative path, plus the cross-path distribution."""
    monte_carlo: MonteCarloReport
    account: AccountStats
    trades: TradeStats
    path_index: int
    initial_cash: float
    manifest: dict = field(default_factory=dict)

    def account_summary(self) -> str:
        a = self.account
        realized_pct = 100.0 * a.ending_realized / self.initial_cash if self.initial_cash else 0.0
        lines = [
            "Account value",
            "",
            f"  starting equity     {_fmt(a.starting_equity)}",
            f"  ending equity       {_fmt(a.ending_equity)}   "
            f"({a.total_return_fraction:+.2%})",
            f"    of which realized {_fmt(a.ending_realized)}   ({realized_pct:+.2f}% of start)",
            f"    of which unreal.  {_fmt(a.ending_unrealized)}",
            "",
            f"  peak equity         {_fmt(a.peak_equity)}",
            f"  trough equity       {_fmt(a.trough_equity)}",
            f"  max drawdown        {_fmt(a.max_drawdown)}   "
            f"({a.max_drawdown_fraction:.2%})",
            f"  time in drawdown    {a.time_in_drawdown_fraction:>13.1%}",
            "",
            f"  peak unrealized     {_fmt(a.peak_unrealized)}",
            f"  trough unrealized   {_fmt(a.trough_unrealized)}",
            f"  peak margin req.    {_fmt(a.peak_margin)}   "
            f"({a.peak_margin_utilization:.1%} of peak equity)",
            f"  max open positions  {a.max_open_positions:>13}",
            "",
            f"  Sharpe (ann.)       {a.sharpe:>13.2f}",
            f"  Sortino (ann.)      {a.sortino:>13.2f}",
            f"  Calmar              {a.calmar:>13.2f}",
            "",
            f"  equity   {a.equity_sparkline()}",
            f"  realized {sparkline(a.realized_curve)}",
            f"  unreal.  {sparkline(a.unrealized_curve)}",
        ]
        return "\n".join(lines)

    def trade_summary(self) -> str:
        t = self.trades
        if t.count == 0:
            return "Trades\n\n  no closed trades"

        def ratio(value: float) -> str:
            return "inf" if value == float("inf") else f"{value:.2f}"

        lines = [
            "Trades",
            "",
            f"  closed trades       {t.count:>13}",
            f"  wins / losses       {t.wins:>6} / {t.losses:<6}"
            f"  win rate {t.win_rate:.1%}",
            f"  scratches           {t.scratches:>13}",
            "",
            f"  total realized      {_fmt(t.total_pnl)}",
            f"  best trade          {_fmt(t.best)}",
            f"  worst trade         {_fmt(t.worst)}",
            f"  mean per trade      {_fmt(t.mean)}",
            f"  median per trade    {_fmt(t.median)}",
            f"  stdev per trade     {_fmt(t.stdev)}",
            "",
            f"  gross profit        {_fmt(t.gross_profit)}",
            f"  gross loss          {_fmt(t.gross_loss)}",
            f"  profit factor       {ratio(t.profit_factor):>14}",
            f"  average win         {_fmt(t.average_win)}",
            f"  average loss        {_fmt(t.average_loss)}",
            f"  payoff ratio        {ratio(t.payoff_ratio):>14}",
            f"  expectancy/trade    {_fmt(t.expectancy)}",
            "",
            f"  skew                {t.skew:>13.2f}",
            f"  excess kurtosis     {t.excess_kurtosis:>13.2f}",
            f"  longest win streak  {t.longest_win_streak:>13}",
            f"  longest loss streak {t.longest_loss_streak:>13}",
            f"  avg holding (days)  {t.average_holding_days:>13.1f}",
            "",
            f"  fees                {_fmt(t.total_fees)}",
            f"  spread cost         {_fmt(t.total_spread_cost)}",
        ]
        percentile_row = "  ".join(f"{k} {v:,.0f}" for k, v in t.percentiles.items())
        lines += ["", f"  percentiles  {percentile_row}"]
        reasons = "  ".join(f"{k} {v}" for k, v in sorted(t.by_close_reason.items()))
        lines += [f"  closed by    {reasons}"]
        return "\n".join(lines)

    def distribution_summary(self) -> str:
        t = self.trades
        if t.count == 0:
            return ""
        lines = [
            "Trade P&L distribution (z-scores: standard deviations from the mean)",
            "",
            t.z_histogram().render(),
            "",
            f"  mean {t.mean:,.2f}   stdev {t.stdev:,.2f}   "
            f"skew {t.skew:+.2f}   excess kurtosis {t.excess_kurtosis:+.2f}",
        ]
        if t.stdev == 0:
            lines += ["", "  All trades produced the same P&L, so z-scores are undefined."]
        else:
            tail = sum(1 for z in t.z_scores if abs(z) > 2.0)
            lines += [
                "",
                f"  {tail} of {t.count} trades beyond 2 sd "
                f"({100.0 * tail / t.count:.1f}%; a normal distribution gives 4.6%)",
            ]
            if t.skew < -0.5:
                lines.append("  Left-skewed: a few large losses carry the result.")
            elif t.skew > 0.5:
                lines.append("  Right-skewed: a few large wins carry the result.")
        lines += ["", "Trade P&L distribution (dollars)", "", t.pnl_histogram().render()]
        return "\n".join(lines)

    def full(self) -> str:
        blocks = [
            self.monte_carlo.summary(),
            self.account_summary(),
            self.trade_summary(),
            self.distribution_summary(),
        ]
        note = (
            "Spread cost is a modeled assumption, not reconstructed execution.\n"
            "Without calibration against real quote history, treat the interval as\n"
            "sensitivity analysis rather than a forecast. Re-run with a different\n"
            "--spread-variance-scale to see how much of the result it owns."
        )
        if self.monte_carlo.spread_model != "zero":
            blocks.append(note)
        separator = "\n\n" + "-" * 72 + "\n\n"
        return separator.join(b for b in blocks if b)


def build_report(result: RunResult, confidence_level: float = 0.95,
                 variance_scale: float = 1.0) -> MonteCarloReport:
    paths = result.paths
    if not paths:
        raise ValueError("no scenarios were run")

    net = sorted(p.net_pnl for p in paths)
    n = len(net)
    mean = statistics.fmean(net)
    stdev = statistics.stdev(net) if n > 1 else 0.0
    stderr = stdev / math.sqrt(n) if n else 0.0
    tail = (1.0 - confidence_level) / 2.0
    mean_spread = statistics.fmean(p.spread_cost for p in paths)

    deterministic = (result.deterministic_pnl
                     if result.deterministic_pnl is not None else mean + mean_spread)

    return MonteCarloReport(
        paths=n,
        seed=result.manifest.spread_mc_seed,
        spread_model=result.manifest.spread_model,
        variance_scale=variance_scale,
        mean_net_pnl=mean,
        median_net_pnl=percentile(net, 0.5),
        stdev_net_pnl=stdev,
        standard_error=stderr,
        ci_low=percentile(net, tail),
        ci_high=percentile(net, 1.0 - tail),
        confidence_level=confidence_level,
        worst_net_pnl=net[0],
        best_net_pnl=net[-1],
        probability_of_profit=sum(1 for x in net if x > 0) / n,
        margin_breach_probability=sum(1 for p in paths if p.margin_breached) / n,
        mean_spread_cost=mean_spread,
        mean_fees=statistics.fmean(p.fees for p in paths),
        mean_max_drawdown=statistics.fmean(p.max_drawdown for p in paths),
        mean_return_fraction=statistics.fmean(p.return_fraction for p in paths),
        deterministic_net_pnl=deterministic,
        separability_residual=deterministic - (mean + mean_spread),
        percentiles={f"p{int(q * 100):02d}": percentile(net, q)
                     for q in (0.05, 0.25, 0.50, 0.75, 0.95)},
        total_fills=sum(p.fill_count for p in paths),
        total_rejections=sum(p.rejection_count for p in paths),
        mean_dividend_cash=statistics.fmean(p.dividend_cash for p in paths),
        total_early_assignments=sum(p.early_assignment_count for p in paths),
        stale_mark_valuations=sum(p.stale_mark_valuations for p in paths),
        max_mark_age_days=max((p.max_mark_age_ns for p in paths), default=0)
        / (86_400 * 1_000_000_000),
        settlements_without_official_price=sum(
            p.settlements_without_official_price for p in paths),
        ledger_reconciles=all(p.ledger_reconciles for p in paths),
        truncated_paths=sum(1 for p in paths if p.truncated),
        quarantined_positions=sum(p.quarantined_positions for p in paths),
        # More than one path but only one outcome: the Monte Carlo said nothing.
        degenerate=n > 1 and stdev == 0.0,
    )


def build_performance_report(result: RunResult, confidence_level: float = 0.95,
                             variance_scale: float = 1.0) -> PerformanceReport:
    """Full report, with trade and account detail from the median path."""
    index = result.representative
    return PerformanceReport(
        monte_carlo=build_report(result, confidence_level, variance_scale),
        account=account_stats(
            result.equity_points[index] if result.equity_points else [],
            result.manifest.initial_cash,
        ),
        trades=trade_stats(result.trades[index] if result.trades else []),
        path_index=index,
        initial_cash=result.manifest.initial_cash,
        manifest=result.manifest.to_dict(),
    )


def convergence_table(results: dict[int, RunResult]) -> str:
    """
    Standard error against path count.

    Monte Carlo error falls as 1/sqrt(n), so quadrupling paths should roughly
    halve it. A table that does not is evidence the draws are correlated.
    """
    lines = [f"{'paths':>8}{'mean P&L':>16}{'stdev':>14}{'std error':>14}"]
    for paths in sorted(results):
        r = build_report(results[paths])
        lines.append(
            f"{r.paths:>8}{r.mean_net_pnl:>16,.2f}{r.stdev_net_pnl:>14,.2f}"
            f"{r.standard_error:>14,.2f}"
        )
    return "\n".join(lines)
