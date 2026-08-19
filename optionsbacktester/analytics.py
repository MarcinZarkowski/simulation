"""
Trade-level and account-level statistics.

Separated from `report.py` because these describe *one path* — the shape of its
trades and the evolution of its account — whereas the Monte Carlo report
describes the distribution *across* paths. Conflating the two is how a reader
ends up thinking a confidence interval says something about individual trades.

Every statistic here is computed from the engine's own trade ledger and equity
series rather than reconstructed from fills, so partial closes, expirations,
exercises and assignments are all counted exactly once.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime

import obt_engine as E

# Trading days per year, for annualizing a per-bar series.
TRADING_DAYS = 252

CLOSE_REASON_NAMES = {
    E.CloseReason.CLOSED: "closed",
    E.CloseReason.EXPIRED: "expired",
    E.CloseReason.EXERCISED: "exercised",
    E.CloseReason.ASSIGNED: "assigned",
    E.CloseReason.ADJUSTED: "adjusted",
}


def _epoch_to_datetime(ns: int) -> datetime:
    return datetime(1970, 1, 1).fromtimestamp(ns / 1e9) if ns else datetime(1970, 1, 1)


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile on an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


@dataclass
class Histogram:
    """Counts per bin, with the bin edges that produced them."""
    edges: list[float]
    counts: list[int]
    label: str = ""

    @property
    def total(self) -> int:
        return sum(self.counts)

    def render(self, width: int = 44, show_edges: bool = True) -> str:
        """
        Terminal bar chart, one row per bin.

        Plain text rather than a plotting library so the report works over ssh,
        in CI logs, and in a terminal with no display.
        """
        if not self.counts or self.total == 0:
            return "  (no trades)"
        peak = max(self.counts)
        lines = []
        for i, count in enumerate(self.counts):
            lo, hi = self.edges[i], self.edges[i + 1]
            bar = "█" * int(round(width * count / peak)) if peak else ""
            share = 100.0 * count / self.total
            label = f"[{lo:>6.2f}, {hi:>6.2f})" if show_edges else ""
            lines.append(f"  {label} {bar:<{width}} {count:>5}  {share:>5.1f}%")
        return "\n".join(lines)


def histogram(values: list[float], bins: int = 13,
              lo: float | None = None, hi: float | None = None,
              label: str = "") -> Histogram:
    """
    Equal-width bins over the value range.

    Bin count defaults to an odd number so a symmetric range puts zero at the
    centre of a bin rather than on an edge, which matters when the quantity being
    binned is a z-score.
    """
    if not values:
        return Histogram([0.0, 0.0], [0], label)
    low = min(values) if lo is None else lo
    high = max(values) if hi is None else hi
    if high <= low:
        high = low + 1.0
    width = (high - low) / bins
    edges = [low + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = int((v - low) / width)
        counts[min(max(idx, 0), bins - 1)] += 1
    return Histogram(edges, counts, label)


@dataclass
class TradeStats:
    """Distribution of realized P&L across individual closed trades."""
    count: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    win_rate: float = 0.0

    total_pnl: float = 0.0
    best: float = 0.0
    worst: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    stdev: float = 0.0

    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    payoff_ratio: float = 0.0

    skew: float = 0.0
    excess_kurtosis: float = 0.0
    percentiles: dict[str, float] = field(default_factory=dict)

    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    average_holding_days: float = 0.0

    total_fees: float = 0.0
    total_spread_cost: float = 0.0
    by_close_reason: dict[str, int] = field(default_factory=dict)

    z_scores: list[float] = field(default_factory=list)
    pnl: list[float] = field(default_factory=list)

    def z_histogram(self, bins: int = 13) -> Histogram:
        """
        Distribution of trade P&L in standard deviations from the mean.

        Standardizing makes the shape comparable across strategies and account
        sizes: a symmetric bell centred on zero means no trade dominated, while a
        long left tail means a few losses carry the result.
        """
        return histogram(self.z_scores, bins=bins, lo=-4.0, hi=4.0,
                         label="trade P&L, standard deviations from mean")

    def pnl_histogram(self, bins: int = 13) -> Histogram:
        return histogram(self.pnl, bins=bins, label="trade P&L, dollars")


def _streaks(pnl: list[float]) -> tuple[int, int]:
    best_win = best_loss = run_win = run_loss = 0
    for value in pnl:
        if value > 0:
            run_win += 1
            run_loss = 0
        elif value < 0:
            run_loss += 1
            run_win = 0
        else:
            run_win = run_loss = 0
        best_win = max(best_win, run_win)
        best_loss = max(best_loss, run_loss)
    return best_win, best_loss


def _moments(values: list[float], mean: float, stdev: float) -> tuple[float, float]:
    """Sample skewness and excess kurtosis; zero when undefined."""
    n = len(values)
    if n < 3 or stdev <= 0:
        return 0.0, 0.0
    m3 = sum((v - mean) ** 3 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    return m3 / stdev**3, m4 / stdev**4 - 3.0


def trade_stats(trades: list[E.TradeRecord]) -> TradeStats:
    """Summarize a path's trade ledger."""
    s = TradeStats()
    if not trades:
        return s

    pnl = [t.realized_pnl for t in trades]
    s.pnl = pnl
    s.count = len(pnl)
    s.wins = sum(1 for v in pnl if v > 0)
    s.losses = sum(1 for v in pnl if v < 0)
    s.scratches = s.count - s.wins - s.losses
    # Scratches are excluded from the denominator: a zero-P&L close is neither a
    # win nor a loss, and counting it as a loss understates the hit rate.
    decided = s.wins + s.losses
    s.win_rate = s.wins / decided if decided else 0.0

    s.total_pnl = sum(pnl)
    s.best = max(pnl)
    s.worst = min(pnl)
    s.mean = statistics.fmean(pnl)
    ordered = sorted(pnl)
    s.median = percentile(ordered, 0.5)
    s.stdev = statistics.stdev(pnl) if s.count > 1 else 0.0

    s.gross_profit = sum(v for v in pnl if v > 0)
    s.gross_loss = -sum(v for v in pnl if v < 0)
    # Undefined with no losses; reported as infinity rather than silently zero.
    s.profit_factor = (s.gross_profit / s.gross_loss) if s.gross_loss > 0 else float("inf")
    s.average_win = s.gross_profit / s.wins if s.wins else 0.0
    s.average_loss = s.gross_loss / s.losses if s.losses else 0.0
    s.payoff_ratio = (s.average_win / s.average_loss) if s.average_loss > 0 else float("inf")
    s.expectancy = s.mean

    s.skew, s.excess_kurtosis = _moments(pnl, s.mean, s.stdev)
    s.percentiles = {
        f"p{int(q * 100):02d}": percentile(ordered, q)
        for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    }
    s.longest_win_streak, s.longest_loss_streak = _streaks(pnl)

    holding = [t.holding_days for t in trades]
    s.average_holding_days = statistics.fmean(holding) if holding else 0.0
    s.total_fees = sum(t.fees for t in trades)
    s.total_spread_cost = sum(t.spread_cost for t in trades)

    for t in trades:
        name = CLOSE_REASON_NAMES.get(t.reason, "unknown")
        s.by_close_reason[name] = s.by_close_reason.get(name, 0) + 1

    s.z_scores = [(v - s.mean) / s.stdev for v in pnl] if s.stdev > 0 else [0.0] * s.count
    return s


@dataclass
class AccountStats:
    """Evolution of the account over the run."""
    points: int = 0
    starting_equity: float = 0.0
    ending_equity: float = 0.0

    # The three series the report plots.
    realized_curve: list[float] = field(default_factory=list)
    unrealized_curve: list[float] = field(default_factory=list)
    total_curve: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    ending_realized: float = 0.0
    ending_unrealized: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_fraction: float = 0.0
    time_in_drawdown_fraction: float = 0.0

    peak_unrealized: float = 0.0
    trough_unrealized: float = 0.0
    peak_margin: float = 0.0
    peak_margin_utilization: float = 0.0
    max_open_positions: int = 0

    total_return_fraction: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0

    def equity_sparkline(self, width: int = 60) -> str:
        return sparkline(self.total_curve, width)


def sparkline(values: list[float], width: int = 60) -> str:
    """
    Single-line trend using block glyphs.

    Downsamples by taking one value per bucket rather than averaging, so a spike
    is not smoothed away.
    """
    if not values:
        return ""
    glyphs = "▁▂▃▄▅▆▇█"
    if len(values) > width:
        step = len(values) / width
        sampled = [values[min(int(i * step), len(values) - 1)] for i in range(width)]
    else:
        sampled = values
    lo, hi = min(sampled), max(sampled)
    if hi <= lo:
        return glyphs[0] * len(sampled)
    span = hi - lo
    return "".join(glyphs[min(int((v - lo) / span * len(glyphs)), len(glyphs) - 1)]
                   for v in sampled)


def account_stats(points: list[E.EquityPoint], initial_cash: float) -> AccountStats:
    """Summarize one path's account evolution."""
    a = AccountStats()
    if not points:
        return a

    a.points = len(points)
    a.realized_curve = [p.realized_pnl for p in points]
    a.unrealized_curve = [p.unrealized_pnl for p in points]
    a.total_curve = [p.equity for p in points]
    a.equity_curve = a.total_curve
    a.starting_equity = initial_cash
    a.ending_equity = a.total_curve[-1]
    a.ending_realized = a.realized_curve[-1]
    a.ending_unrealized = a.unrealized_curve[-1]

    a.peak_equity = max(a.total_curve)
    a.trough_equity = min(a.total_curve)
    a.peak_unrealized = max(a.unrealized_curve)
    a.trough_unrealized = min(a.unrealized_curve)
    a.peak_margin = max(p.margin_requirement for p in points)
    a.max_open_positions = max(p.open_positions for p in points)

    # Drawdown measured against the running peak, which is what an account
    # holder actually experiences, not against the starting balance.
    running_peak = initial_cash
    under_water = 0
    for value in a.total_curve:
        running_peak = max(running_peak, value)
        drop = running_peak - value
        a.max_drawdown = max(a.max_drawdown, drop)
        if drop > 0:
            under_water += 1
            if running_peak > 0:
                a.max_drawdown_fraction = max(a.max_drawdown_fraction, drop / running_peak)
    a.time_in_drawdown_fraction = under_water / len(a.total_curve)

    if initial_cash:
        a.total_return_fraction = (a.ending_equity - initial_cash) / initial_cash
    if a.peak_equity > 0:
        a.peak_margin_utilization = a.peak_margin / a.peak_equity

    # Per-bar simple returns. The bar interval is whatever the feed supplies, so
    # these are annualized on a 252-day basis and are comparable only between runs
    # on the same resolution.
    returns = [
        (b - a_) / a_ for a_, b in zip(a.total_curve, a.total_curve[1:], strict=False) if a_
    ]
    if len(returns) > 1:
        mean_r = statistics.fmean(returns)
        sd = statistics.stdev(returns)
        if sd > 0:
            a.sharpe = mean_r / sd * math.sqrt(TRADING_DAYS)
        downside = [r for r in returns if r < 0]
        if len(downside) > 1:
            dsd = statistics.stdev(downside)
            if dsd > 0:
                a.sortino = mean_r / dsd * math.sqrt(TRADING_DAYS)
    if a.max_drawdown > 0:
        a.calmar = (a.ending_equity - initial_cash) / a.max_drawdown
    return a
