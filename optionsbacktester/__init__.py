"""Options strategy backtester over the OptionsBackfill data lake."""
from .analytics import (
    AccountStats,
    Histogram,
    TradeStats,
    account_stats,
    histogram,
    sparkline,
    trade_stats,
)
from .report import (
    MonteCarloReport,
    PerformanceReport,
    build_performance_report,
    build_report,
    convergence_table,
)
from .runner import RunManifest, RunResult, run
from .stream import DataLake, DaySlice, UniverseFilter, iter_days, load_day
from .strategy import Chain, ChainRow, Context, Strategy, buy, group, sell

__all__ = [
    "AccountStats", "Chain", "ChainRow", "Context", "DataLake", "DaySlice",
    "Histogram", "MonteCarloReport", "PerformanceReport", "RunManifest",
    "RunResult", "Strategy", "TradeStats", "UniverseFilter", "account_stats",
    "build_performance_report", "build_report", "buy", "convergence_table",
    "group", "histogram", "iter_days", "load_day", "run", "sell", "sparkline",
    "trade_stats",
]
