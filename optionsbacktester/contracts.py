"""
Translate pipeline rows into engine structs.

The pipeline is the source of truth for contract terms, marks, and analytics
quality. Nothing here recomputes a price or invents a strike: the mapping is
mechanical, and where the pipeline says a row is unusable the flag is carried
through rather than reinterpreted.
"""
from __future__ import annotations

from datetime import date, datetime

import polars as pl

import obt_engine as E

NS_PER_SECOND = 1_000_000_000


def to_ns(value: datetime | date) -> int:
    """
    Epoch nanoseconds from a tz-naive UTC datetime, matching the lake.

    Accepts a plain ``date`` too, at midnight: the corporate-actions frame stores
    ex-dates and pay dates as dates, and midnight UTC precedes every session open,
    so a dividend lands on the first bar of its ex-date.
    """
    if not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
    if value.tzinfo is not None:
        value = value.replace(tzinfo=None)
    epoch = datetime(1970, 1, 1)
    delta = value - epoch
    return (delta.days * 86400 + delta.seconds) * NS_PER_SECOND + delta.microseconds * 1000


def contract_version_key(symbol: str, strike: float, equity_amount: float, multiplier: float) -> int:
    """
    Identity for one set of terms.

    Deliberately includes the deliverable and the multiplier, not just the
    symbol: after an adjustment the same symbol can describe different
    economics, and treating those as one instrument is the bug this avoids.
    """
    return E.hash_symbol(f"{symbol}|{strike:.6f}|{equity_amount:.6f}|{multiplier:.6f}")


def instrument_key(underlying: str, expiration: datetime, flag: str, strike: float) -> int:
    """Identity for the economic series, which survives a symbol change."""
    return E.hash_symbol(f"{underlying}|{expiration.isoformat()}|{flag}|{strike:.6f}")


def _float(row: dict, name: str, default: float) -> float:
    value = row.get(name)
    return default if value is None else float(value)


_PROVENANCE = {
    "point_in_time_snapshot": E.TermsProvenance.POINT_IN_TIME,
    "later_snapshot_backfilled": E.TermsProvenance.BACKFILLED,
}


def _version_timing(versions: pl.DataFrame) -> dict[int, tuple[int, int, object]]:
    """
    ``(valid_from, source_available_at, terms_provenance)`` per version key, from
    ``option_contract_version.parquet``.

    Keyed on the same terms tuple the engine uses rather than on the pipeline's own
    ``contract_version_id``, because the two are independent hashes over different
    string encodings and will not agree.
    """
    needed = ("symbol", "strike", "deliverable_equity_amount", "quote_multiplier")
    if versions.is_empty() or any(c not in versions.columns for c in needed):
        return {}

    out: dict[int, tuple[int, int, object]] = {}
    for row in versions.iter_rows(named=True):
        key = contract_version_key(
            row["symbol"], _float(row, "strike", 0.0),
            _float(row, "deliverable_equity_amount", 100.0),
            _float(row, "quote_multiplier", 100.0),
        )
        valid_from = row.get("valid_from")
        available = row.get("source_available_at")
        out[key] = (
            to_ns(valid_from) if valid_from else 0,
            to_ns(available) if available else 0,
            _PROVENANCE.get(row.get("terms_provenance"), E.TermsProvenance.UNKNOWN),
        )
    return out


def build_contracts(
    options: pl.DataFrame,
    underlying_symbol: str,
    versions: pl.DataFrame | None = None,
) -> dict[int, E.OptionContractVersion]:
    """
    One contract version per distinct set of terms seen in the frame.

    Deduplicated on the version key, so a contract quoted across 390 minutes
    produces one struct rather than 390.

    ``versions`` is the pipeline's ``option_contract_version`` frame. It supplies
    the point-in-time fields the bars frame does not carry -- when terms took
    effect, when they became knowable, and whether they were observed at the time
    or copied from a later snapshot. Without it every version is un-provenanced,
    and the engine refuses to open a position on an adjusted contract whose terms
    it cannot establish.
    """
    if options.is_empty():
        return {}
    timing = _version_timing(versions if versions is not None else pl.DataFrame())

    wanted = [
        "symbol", "strike", "pricing_strike", "flag", "expiration",
        "quote_multiplier", "deliverable_equity_amount", "deliverable_cash_amount",
        "is_adjusted_contract", "adjusted_pricing_status",
    ]
    present = [c for c in wanted if c in options.columns]

    # Deduplicate on the terms that define a version, not on the symbol. The same
    # OCC symbol either side of an adjustment describes different economics, and
    # collapsing those onto one version is exactly what contract_version_key
    # exists to prevent: build_bars derives the key from each row's own terms, so
    # a dropped variant's bars would silently fail to match and disappear.
    version_columns = [
        c for c in ("symbol", "strike", "deliverable_equity_amount", "quote_multiplier")
        if c in present
    ]
    unique_terms = options.select(present).unique(subset=version_columns, keep="first")

    out: dict[int, E.OptionContractVersion] = {}
    for row in unique_terms.iter_rows(named=True):
        symbol = row["symbol"]
        strike = _float(row, "strike", 0.0)
        multiplier = _float(row, "quote_multiplier", 100.0)
        equity_amount = _float(row, "deliverable_equity_amount", 100.0)
        expiration = row["expiration"]

        c = E.OptionContractVersion()
        key = contract_version_key(symbol, strike, equity_amount, multiplier)
        c.id = key
        c.instrument_id = instrument_key(underlying_symbol, expiration, row["flag"], strike)
        c.symbol = symbol
        c.underlying_symbol = underlying_symbol
        c.type = E.OptionType.CALL if row["flag"] == "c" else E.OptionType.PUT
        c.strike = strike
        c.pricing_strike = _float(row, "pricing_strike", strike)
        c.quote_multiplier = int(round(multiplier))
        c.deliverable_equity_microshares = int(round(equity_amount * 1_000_000))
        c.deliverable_cash = _float(row, "deliverable_cash_amount", 0.0)
        c.is_adjusted = bool(row.get("is_adjusted_contract", False))

        status = row.get("adjusted_pricing_status") or "standard_contract"
        # The pipeline refuses to price deliverables it cannot value exactly.
        # Such a contract may be exited but never entered.
        priced = status != "unpriced_adjusted_contract"
        c.tradable_for_new_positions = priced
        c.analytics_supported = priced

        c.expiration = to_ns(expiration)
        # From the reference frame when we have it. The fallback is the epoch,
        # which asserts the terms held forever and were always knowable -- true
        # enough for an unadjusted contract, and refused for an adjusted one by
        # the provenance gate rather than by silently trusting it.
        valid_from, available_at, provenance = timing.get(
            key, (0, 0, E.TermsProvenance.UNKNOWN))
        c.valid_from = valid_from
        c.source_available_at = available_at
        c.terms_provenance = provenance
        c.valid_to = c.expiration
        out[key] = c
    return out


def build_bars(batch: pl.DataFrame, contracts: dict[int, E.OptionContractVersion]) -> list[E.MarketBar]:
    """Market bars for one timestamp."""
    bars: list[E.MarketBar] = []
    for row in batch.iter_rows(named=True):
        key = contract_version_key(
            row["symbol"], _float(row, "strike", 0.0),
            _float(row, "deliverable_equity_amount", 100.0),
            _float(row, "quote_multiplier", 100.0),
        )
        if key not in contracts:
            continue
        b = E.MarketBar()
        b.timestamp = to_ns(row["timestamp"])
        b.contract_version_id = key
        b.open = _float(row, "open", 0.0)
        b.high = _float(row, "high", 0.0)
        b.low = _float(row, "low", 0.0)
        b.close = _float(row, "close", 0.0)
        b.vwap = _float(row, "vwap", 0.0)
        # The pipeline's chosen mark. Falling back to close keeps a bar usable
        # when only the close is populated.
        b.valuation_price = _float(row, "valuation_price", b.close)
        b.volume = int(row.get("volume") or 0)
        b.trade_count = int(row.get("trade_count") or 0)
        b.stale = bool(row.get("is_stale", False))
        # Analytics are valid only if the pipeline solved a real IV for them.
        b.analytics_valid = not (
            bool(row.get("iv_failed", False)) or bool(row.get("iv_is_model_fallback", False))
        )
        bars.append(b)
    return bars


def build_analytics(batch: pl.DataFrame, contracts: dict[int, E.OptionContractVersion]) -> list[E.OptionAnalytics]:
    """Greeks and IV for one timestamp, marked invalid where the pipeline fell back."""
    out: list[E.OptionAnalytics] = []
    for row in batch.iter_rows(named=True):
        key = contract_version_key(
            row["symbol"], _float(row, "strike", 0.0),
            _float(row, "deliverable_equity_amount", 100.0),
            _float(row, "quote_multiplier", 100.0),
        )
        if key not in contracts:
            continue
        a = E.OptionAnalytics()
        a.timestamp = to_ns(row["timestamp"])
        a.contract_version_id = key
        a.implied_volatility = _float(row, "smoothed_iv", 0.0)
        a.delta = _float(row, "delta", 0.0)
        a.gamma = _float(row, "gamma", 0.0)
        a.theta = _float(row, "theta", 0.0)
        a.vega = _float(row, "vega", 0.0)
        a.rho = _float(row, "rho", 0.0)
        a.valid = not (
            bool(row.get("iv_failed", False)) or bool(row.get("iv_is_model_fallback", False))
        )
        out.append(a)
    return out


def build_equity_bars(stock: pl.DataFrame, underlying_symbol: str) -> list[E.EquityBar]:
    """
    Share bars for one timestamp, from the pipeline's stock frame.

    The frame was loaded into ``DaySlice`` and used only to join an underlying price
    onto the option rows, so its open, high, low and volume never reached the
    engine and an equity order had no price to execute against.
    """
    if stock.is_empty():
        return []
    out: list[E.EquityBar] = []
    for row in stock.iter_rows(named=True):
        b = E.EquityBar()
        b.timestamp = to_ns(row["timestamp"])
        b.symbol = underlying_symbol
        close = _float(row, "underlying_close", 0.0)
        b.close = close
        # An absent open means the bar carried only a close. Falling back keeps the
        # bar usable rather than producing a zero price that would be refused.
        b.open = _float(row, "underlying_open", close)
        b.high = _float(row, "underlying_high", close)
        b.low = _float(row, "underlying_low", close)
        b.vwap = _float(row, "underlying_vwap", close)
        b.volume = int(row.get("underlying_volume") or 0)
        b.trade_count = int(row.get("underlying_trade_count") or 0)
        out.append(b)
    return out


def build_bar_view(
    timestamp: datetime,
    batch: pl.DataFrame,
    contracts: dict[int, E.OptionContractVersion],
    underlying_symbol: str,
    stock: pl.DataFrame | None = None,
):
    """
    Snapshot and chain from ONE pass over the batch.

    ``build_bars``, ``build_analytics`` and ``chain_from_batch`` each iterated the
    same frame independently, so every bar was materialized into Python dicts three
    times and the version key recomputed three times per row. They remain available
    separately because tests and callers use them individually; this is the path the
    runner takes.
    """
    from .strategy import Chain, chain_row_from

    snap = E.MarketSnapshot()
    snap.timestamp = to_ns(timestamp)
    if stock is not None:
        snap.equity_bars = build_equity_bars(stock, underlying_symbol)

    bars: list[E.MarketBar] = []
    analytics: list[E.OptionAnalytics] = []
    chain_rows = []
    underlying: float | None = None

    for row in batch.iter_rows(named=True):
        key = contract_version_key(
            row["symbol"], _float(row, "strike", 0.0),
            _float(row, "deliverable_equity_amount", 100.0),
            _float(row, "quote_multiplier", 100.0),
        )
        contract = contracts.get(key)
        if contract is None:
            continue

        price = row.get("underlying_price")
        if price is not None:
            underlying = float(price)

        timestamp_ns = to_ns(row["timestamp"])
        # One validity decision, used by both the bar and the analytics row.
        analytics_valid = not (
            bool(row.get("iv_failed", False)) or bool(row.get("iv_is_model_fallback", False))
        )

        bars.append(_market_bar(row, key, timestamp_ns, analytics_valid))
        analytics.append(_option_analytics(row, key, timestamp_ns, analytics_valid))
        if contract.analytics_supported:
            chain_rows.append(chain_row_from(row, key))

    snap.bars = bars
    snap.analytics = analytics
    snap.underlying_price = {underlying_symbol: underlying} if underlying is not None else {}
    return snap, Chain(chain_rows, underlying)


def _market_bar(row: dict, key: int, timestamp_ns: int, analytics_valid: bool) -> E.MarketBar:
    b = E.MarketBar()
    b.timestamp = timestamp_ns
    b.contract_version_id = key
    b.open = _float(row, "open", 0.0)
    b.high = _float(row, "high", 0.0)
    b.low = _float(row, "low", 0.0)
    b.close = _float(row, "close", 0.0)
    b.vwap = _float(row, "vwap", 0.0)
    b.valuation_price = _float(row, "valuation_price", b.close)
    b.volume = int(row.get("volume") or 0)
    b.trade_count = int(row.get("trade_count") or 0)
    b.stale = bool(row.get("is_stale", False))
    b.analytics_valid = analytics_valid
    return b


def _option_analytics(row: dict, key: int, timestamp_ns: int,
                      analytics_valid: bool) -> E.OptionAnalytics:
    a = E.OptionAnalytics()
    a.timestamp = timestamp_ns
    a.contract_version_id = key
    a.implied_volatility = _float(row, "smoothed_iv", 0.0)
    a.delta = _float(row, "delta", 0.0)
    a.gamma = _float(row, "gamma", 0.0)
    a.theta = _float(row, "theta", 0.0)
    a.vega = _float(row, "vega", 0.0)
    a.rho = _float(row, "rho", 0.0)
    a.valid = analytics_valid
    return a


def build_snapshot(
    timestamp: datetime,
    batch: pl.DataFrame,
    contracts: dict[int, E.OptionContractVersion],
    underlying_symbol: str,
    stock: pl.DataFrame | None = None,
) -> E.MarketSnapshot:
    """A full point-in-time snapshot for the engine."""
    snap = E.MarketSnapshot()
    snap.timestamp = to_ns(timestamp)
    snap.bars = build_bars(batch, contracts)
    snap.analytics = build_analytics(batch, contracts)
    if stock is not None:
        snap.equity_bars = build_equity_bars(stock, underlying_symbol)

    price = None
    if "underlying_price" in batch.columns:
        prices = batch["underlying_price"].drop_nulls()
        if len(prices) > 0:
            price = float(prices[0])
    snap.underlying_price = {underlying_symbol: price} if price is not None else {}
    return snap


def build_dividends(
    corporate_actions: pl.DataFrame,
    underlying_symbol: str,
) -> list[E.DividendEvent]:
    """
    Cash dividends from the pipeline's ``corporate_actions`` frame.

    ``corporate_actions`` was loaded into ``DaySlice`` and never read, so a
    dividend did three things it should not: it was never paid on a share
    position, so a covered call understated its return by the whole yield; it
    never triggered the early assignment a call holder would rationally take; and
    the declaration date the pipeline gates on was ignored.

    ``declared_at`` takes the later of the company's declaration and the moment the
    source made it available. Either alone would let a backtest anticipate the
    announcement -- the first because the vendor had not published it yet, the
    second because a vendor backfill can predate the announcement it describes.
    """
    if corporate_actions.is_empty() or "type" not in corporate_actions.columns:
        return []

    rows = corporate_actions.filter(pl.col("type") == "dividend")
    out: list[E.DividendEvent] = []
    for row in rows.iter_rows(named=True):
        amount = row.get("amount")
        ex_date = row.get("date")
        if not amount or not ex_date:
            continue
        d = E.DividendEvent()
        d.underlying_symbol = underlying_symbol
        d.amount_per_share = float(amount)
        d.ex_date = to_ns(ex_date)
        # Unstated pay date means same-day, which understates the float but never
        # invents cash.
        pay = row.get("pay_date")
        d.pay_date = to_ns(pay) if pay else d.ex_date
        declared = [to_ns(v) for v in (row.get("declared_date"), row.get("source_available_at")) if v]
        d.declared_at = max(declared) if declared else d.ex_date
        out.append(d)
    return out


def build_lineage_transitions(
    lineage: pl.DataFrame,
    contracts: dict[int, E.OptionContractVersion],
) -> list[E.CorporateActionTransition]:
    """
    Convert pipeline lineage rows into engine transitions.

    ``occ_confirmed`` and the quantity conversion are carried through verbatim.
    An unconfirmed row still becomes a transition so the engine can refuse to
    carry a position through it, rather than the adjustment passing unnoticed.
    """
    if lineage.is_empty():
        return []

    by_symbol = {c.symbol: key for key, c in contracts.items()}
    out: list[E.CorporateActionTransition] = []
    for row in lineage.iter_rows(named=True):
        t = E.CorporateActionTransition()
        t.lineage_event_id = E.hash_symbol(str(row.get("lineage_event_id", "")))
        effective = row.get("effective_at")
        available = row.get("source_available_at")
        t.effective_at = to_ns(effective) if effective else 0
        t.source_available_at = to_ns(available) if available else t.effective_at
        t.parent_version_id = by_symbol.get(row.get("parent_symbol"), 0)
        t.child_version_id = by_symbol.get(row.get("child_symbol"), 0)
        t.parent_contracts = int(row.get("parent_contracts") or 0)
        t.child_contracts = int(row.get("child_contracts") or 0)
        t.occ_confirmed = bool(row.get("occ_confirmed", False))
        out.append(t)
    return out
