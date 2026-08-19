"""
Streaming reader for the OptionsBackfill data lake.

The lake is laid out as ``data/{TICKER}/{YYYY}/{MM}/{DD}/`` with one file per
kind per day. Only the day directories inside the requested range are opened, and
only the columns the engine consumes are read, so peak memory is one day of one
ticker rather than the whole history.

Nothing here interprets prices. Quality gating is expressed as filters over the
pipeline's own flags, so the engine and the pipeline never disagree about which
rows are usable.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import polars as pl

# Columns the engine actually needs. Projecting explicitly keeps a wide
# 49-column file from being read in full.
OPTION_COLUMNS = [
    "symbol", "timestamp", "open", "high", "low", "close", "vwap",
    "valuation_price", "volume", "trade_count", "underlying_price",
    "strike", "pricing_strike", "flag", "expiration",
    "quote_multiplier", "deliverable_equity_amount", "deliverable_cash_amount",
    "is_adjusted_contract", "adjusted_pricing_status",
    "is_stale", "iv_failed", "iv_is_model_fallback",
    "smoothed_iv", "delta", "gamma", "theta", "vega", "rho", "theoretical_value",
]

# vwap and trade_count are written by the pipeline and were not read. A strategy
# trading shares needs them for the same reasons the option side does: a
# consistent mark, and a way to tell a real bar from a printless one.
STOCK_COLUMNS = ["timestamp", "underlying_open", "underlying_high", "underlying_low",
                 "underlying_close", "underlying_vwap", "underlying_volume",
                 "underlying_trade_count"]

SUCCESS_MARKER = "_SUCCESS"


@dataclass(frozen=True)
class UniverseFilter:
    """
    Which contracts a strategy is allowed to see.

    Applied as predicates during the scan so excluded rows are never
    materialized. Quality gates default to on: a row whose IV came from a model
    fallback, or whose bar is stale, is not something to trade against.
    """
    option_types: tuple[str, ...] = ("c", "p")
    min_dte: int | None = None
    max_dte: int | None = None
    min_abs_delta: float | None = None
    max_abs_delta: float | None = None
    min_volume: int | None = None
    min_moneyness: float | None = None
    max_moneyness: float | None = None
    symbols: tuple[str, ...] | None = None

    exclude_stale: bool = True
    exclude_failed_iv: bool = True
    exclude_fallback_iv: bool = True
    exclude_adjusted: bool = False
    # Contracts the pipeline could not price are never tradable for new
    # positions; keeping them visible only helps a strategy exit one.
    exclude_unpriced_adjusted: bool = True

    def predicates(self) -> list[pl.Expr]:
        out: list[pl.Expr] = []
        if set(self.option_types) != {"c", "p"}:
            out.append(pl.col("flag").is_in(list(self.option_types)))
        if self.symbols:
            out.append(pl.col("symbol").is_in(list(self.symbols)))
        if self.exclude_stale:
            out.append(~pl.col("is_stale").fill_null(True))
        if self.exclude_failed_iv:
            out.append(~pl.col("iv_failed").fill_null(True))
        if self.exclude_fallback_iv:
            out.append(~pl.col("iv_is_model_fallback").fill_null(True))
        if self.exclude_adjusted:
            out.append(~pl.col("is_adjusted_contract").fill_null(False))
        if self.exclude_unpriced_adjusted:
            out.append(pl.col("adjusted_pricing_status") != "unpriced_adjusted_contract")
        if self.min_volume is not None:
            out.append(pl.col("volume").fill_null(0) >= self.min_volume)
        return out

    def derived_predicates(self) -> list[pl.Expr]:
        """Predicates on columns computed after the scan."""
        out: list[pl.Expr] = []
        if self.min_dte is not None:
            out.append(pl.col("dte") >= self.min_dte)
        if self.max_dte is not None:
            out.append(pl.col("dte") <= self.max_dte)
        # Delta bounds are per share, matching how strikes are selected.
        if self.min_abs_delta is not None:
            out.append(pl.col("delta_per_share").abs() >= self.min_abs_delta)
        if self.max_abs_delta is not None:
            out.append(pl.col("delta_per_share").abs() <= self.max_abs_delta)
        if self.min_moneyness is not None:
            out.append(pl.col("moneyness") >= self.min_moneyness)
        if self.max_moneyness is not None:
            out.append(pl.col("moneyness") <= self.max_moneyness)
        return out


@dataclass
class DaySlice:
    """One trading day of option bars plus the underlying, already filtered."""
    day: date
    ticker: str
    options: pl.DataFrame
    stock: pl.DataFrame
    corporate_actions: pl.DataFrame = field(default_factory=pl.DataFrame)
    contract_versions: pl.DataFrame = field(default_factory=pl.DataFrame)
    lineage_events: pl.DataFrame = field(default_factory=pl.DataFrame)

    def timestamps(self) -> list[datetime]:
        if self.options.is_empty():
            return []
        return self.options["timestamp"].unique().sort().to_list()


class DataLake:
    """Locates and reads the partitioned lake for one ticker."""

    def __init__(self, root: str | Path, ticker: str):
        self.root = Path(root)
        self.ticker = ticker
        self.ticker_dir = self.root / ticker

    def day_dirs(self, start: date | None = None, end: date | None = None) -> list[Path]:
        """
        Day directories in range, in chronological order.

        Walking the YYYY/MM/DD tree and pruning by directory name means a
        one-week backtest over five years of history opens five directories, not
        the whole tree.
        """
        if not self.ticker_dir.is_dir():
            return []
        out: list[Path] = []
        for year_dir in sorted(self.ticker_dir.glob("[0-9][0-9][0-9][0-9]")):
            year = int(year_dir.name)
            if start and year < start.year:
                continue
            if end and year > end.year:
                continue
            for month_dir in sorted(year_dir.glob("[0-9][0-9]")):
                month = int(month_dir.name)
                if start and (year, month) < (start.year, start.month):
                    continue
                if end and (year, month) > (end.year, end.month):
                    continue
                for day_dir in sorted(month_dir.glob("[0-9][0-9]")):
                    try:
                        d = date(year, month, int(day_dir.name))
                    except ValueError:
                        continue
                    if start and d < start:
                        continue
                    if end and d > end:
                        continue
                    out.append(day_dir)
        return out

    def complete_day_dirs(self, start: date | None = None, end: date | None = None) -> list[Path]:
        """Only days the pipeline marked complete, so a partial day cannot be traded."""
        return [d for d in self.day_dirs(start, end) if (d / SUCCESS_MARKER).exists()]

    @staticmethod
    def day_of(day_dir: Path) -> date:
        return date(int(day_dir.parent.parent.name), int(day_dir.parent.name), int(day_dir.name))

    @staticmethod
    def manifest(day_dir: Path) -> dict:
        marker = day_dir / SUCCESS_MARKER
        if not marker.exists():
            return {}
        try:
            return json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            return {}


# Available-column sets, keyed by (lake directory, file name).
#
# Reading a file's schema means reading its Parquet metadata, which cost ~0.8 ms and
# happened twice per trading day -- 15% of a run's wall clock on a 40-day lake, spent
# re-establishing that every day's options file has the same columns as the last.
#
# Cached per (lake, file name) rather than per path, so a 252-day run pays it twice
# instead of 504 times. A schema that genuinely changes mid-history is handled by the
# fallback below rather than by giving up the cache: a select against a stale set
# raises, and the entry is then re-derived for that file. So the cache is a
# performance assumption, not a correctness one.
_SCHEMA_CACHE: dict[tuple[str, str], frozenset[str]] = {}


def _available_columns(path: Path, *, refresh: bool = False) -> frozenset[str]:
    key = (str(path.parent.parent.parent.parent), path.name)
    if refresh or key not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[key] = frozenset(pl.scan_parquet(path).collect_schema().names())
    return _SCHEMA_CACHE[key]


def _collect_projected(path: Path, columns: list[str], transform=None) -> pl.DataFrame:
    """
    Project `path` to `columns`, apply `transform`, and collect.

    The retry wraps the COLLECT, not the select. Polars is lazy, so a projection
    naming a column the file does not have raises at collection time -- a try around
    the select would never fire, and the cache would be a correctness assumption
    after all.
    """
    for refresh in (False, True):
        available = _available_columns(path, refresh=refresh)
        lazy = pl.scan_parquet(path).select([c for c in columns if c in available])
        if transform is not None:
            lazy = transform(lazy)
        try:
            return lazy.collect()
        except pl.exceptions.ColumnNotFoundError:
            if refresh:
                raise
    raise AssertionError("unreachable")


def _read_optional(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    if not columns:
        return pl.scan_parquet(path).collect()
    return _collect_projected(path, columns)


def load_day(
    day_dir: Path,
    ticker: str,
    universe: UniverseFilter | None = None,
) -> DaySlice:
    """
    Read and filter one day.

    ``dte`` and ``moneyness`` are derived here rather than read, because the
    pipeline deliberately does not store time-to-expiry: recomputing it from
    ``expiration - timestamp`` keeps it exact at every bar.
    """
    universe = universe or UniverseFilter()
    options_path = day_dir / "options_enriched.parquet"
    if not options_path.exists():
        return DaySlice(DataLake.day_of(day_dir), ticker, pl.DataFrame(), pl.DataFrame())

    def shape(scan: pl.LazyFrame) -> pl.LazyFrame:
        for predicate in universe.predicates():
            scan = scan.filter(predicate)
        options = scan.with_columns(
            ((pl.col("expiration") - pl.col("timestamp")).dt.total_seconds() / 86400.0)
            .alias("dte"),
            pl.when(pl.col("underlying_price") > 0)
            .then((pl.col("strike") / pl.col("underlying_price")).log())
            .otherwise(None)
            .alias("moneyness"),
            # The pipeline scales Greeks per 100-share contract; selection is per share.
            (pl.col("delta") / pl.col("deliverable_equity_amount").fill_null(100.0))
            .alias("delta_per_share"),
        )
        for predicate in universe.derived_predicates():
            options = options.filter(predicate)
        return options.sort(["timestamp", "symbol"])

    return DaySlice(
        day=DataLake.day_of(day_dir),
        ticker=ticker,
        options=_collect_projected(options_path, OPTION_COLUMNS, shape),
        stock=_read_optional(day_dir / "stock.parquet", STOCK_COLUMNS),
        corporate_actions=_read_optional(day_dir / "corporate_actions.parquet"),
        contract_versions=_read_optional(day_dir / "option_contract_version.parquet"),
        lineage_events=_read_optional(day_dir / "option_lineage_event.parquet"),
    )


def iter_days(
    lake: DataLake,
    start: date | None = None,
    end: date | None = None,
    universe: UniverseFilter | None = None,
    require_complete: bool = True,
) -> Iterator[DaySlice]:
    """
    Yield one day at a time in chronological order.

    A generator rather than a list: only the day being processed is resident, so
    a multi-year run has flat memory.
    """
    dirs = lake.complete_day_dirs(start, end) if require_complete else lake.day_dirs(start, end)
    for day_dir in dirs:
        day = load_day(day_dir, lake.ticker, universe)
        if not day.options.is_empty():
            yield day


def iter_timestamp_batches(day: DaySlice) -> Iterator[tuple[datetime, pl.DataFrame]]:
    """
    Split a day into per-timestamp batches.

    The engine is called once per timestamp, never once per option row: a single
    SPY minute can carry thousands of contracts, and crossing into Python for
    each would dominate runtime.
    """
    if day.options.is_empty():
        return
    for (ts,), batch in day.options.group_by(["timestamp"], maintain_order=True):
        yield ts, batch
