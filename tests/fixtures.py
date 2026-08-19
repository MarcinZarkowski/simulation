"""
Deterministic synthetic data lake in the pipeline's exact output format.

Written to the real 49-column ``options_enriched`` schema, with the real
directory layout and ``_SUCCESS`` markers, so the reader and the engine are
exercised against the format they will actually meet rather than a convenient
stand-in.

Everything is a closed-form function of the day index: no randomness, no clock,
no network. The same call always produces byte-identical files, which is what
lets a golden end-to-end test assert exact ledger values.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

# The pipeline's canonical column order. Kept literal rather than imported so a
# schema change in the pipeline shows up here as a test failure instead of
# silently propagating.
OPTIONS_ENRICHED_SCHEMA = [
    "symbol", "timestamp", "stock_timestamp", "open", "high", "low", "close", "vwap",
    "valuation_price", "valuation_price_source", "iv_input_type", "volume", "trade_count",
    "underlying_price", "pricing_underlying_price", "strike", "pricing_strike",
    "pricing_valuation_price", "flag", "expiration",
    "T", "r", "rate_source", "q", "contract_multiplier", "quote_multiplier",
    "deliverable_equity_amount", "deliverable_cash_amount", "deliverable_count",
    "deliverable_status", "adjusted_root", "standard_underlying", "adjusted_pricing_status",
    "is_adjusted_contract",
    "is_stale", "dt_stock_min", "iv_failed", "iv_status", "iv_is_model_fallback",
    "pricing_model", "model_validation_status",
    "smoothed_iv", "theoretical_price", "theoretical_value",
    "delta", "gamma", "theta", "vega", "rho",
]

STOCK_SCHEMA = [
    "timestamp", "underlying_open", "underlying_high", "underlying_low",
    "underlying_close", "underlying_vwap", "underlying_volume", "underlying_trade_count",
]

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def black_scholes(S: float, K: float, T: float, r: float, v: float, is_call: bool) -> dict:
    """
    Closed-form price and Greeks, scaled the way the pipeline scales them:
    price per share, value per contract, Greeks per 100-share contract, theta per
    calendar day, vega and rho per one percentage point.
    """
    if T <= 0 or v <= 0 or S <= 0 or K <= 0:
        intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
        return {"price": intrinsic, "delta": (1.0 if intrinsic > 0 else 0.0) * (1 if is_call else -1),
                "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    sqrt_t = v * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * v * v) * T) / sqrt_t
    d2 = d1 - sqrt_t
    disc = math.exp(-r * T)
    pdf = _norm_pdf(d1)

    if is_call:
        price = S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        rho = K * T * disc * _norm_cdf(d2) / 100.0
        theta_yr = -(S * pdf * v / (2.0 * math.sqrt(T)) + r * K * disc * _norm_cdf(d2))
    else:
        price = K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = -_norm_cdf(-d1)
        rho = -K * T * disc * _norm_cdf(-d2) / 100.0
        theta_yr = -(S * pdf * v / (2.0 * math.sqrt(T)) - r * K * disc * _norm_cdf(-d2))

    return {
        "price": price,
        "delta": delta * 100.0,
        "gamma": (pdf / (S * sqrt_t)) * 100.0,
        "theta": (theta_yr / 365.25) * 100.0,
        "vega": (S * pdf * math.sqrt(T) / 100.0) * 100.0,
        "rho": rho * 100.0,
    }


def occ_symbol(root: str, expiration: date, flag: str, strike: float) -> str:
    """Standard 21-character OCC symbol."""
    return f"{root}{expiration:%y%m%d}{flag.upper()}{int(round(strike * 1000)):08d}"


@dataclass
class LakeSpec:
    """
    Shape of the synthetic lake.

    ``underlying_path`` maps a day index to a price, so a test can pick a ramp,
    a crash, or a flat tape and know exactly what the strategy will see.
    """
    ticker: str = "TEST"
    start: date = date(2024, 1, 2)
    trading_days: int = 30
    bars_per_day: int = 3
    strikes: tuple[float, ...] = (80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0)
    # Days to expiry, measured from the first day. A long tenor is included so a
    # poor man's covered call has a LEAP to buy.
    expiry_offsets: tuple[int, ...] = (30, 60, 400)
    risk_free_rate: float = 0.05
    implied_vol: float = 0.25
    volume_per_bar: int = 500
    underlying_path: object = None
    # Rows deliberately marked unusable, so quality gating can be tested.
    stale_symbols: tuple[str, ...] = ()
    fallback_iv_symbols: tuple[str, ...] = ()
    unpriced_adjusted_symbols: tuple[str, ...] = ()
    extra_columns: dict = field(default_factory=dict)

    def price_on(self, day_index: int) -> float:
        if self.underlying_path is None:
            return 100.0
        if callable(self.underlying_path):
            return float(self.underlying_path(day_index))
        return float(self.underlying_path[day_index])


def _trading_days(start: date, count: int) -> list[date]:
    """Weekdays only. Holidays are irrelevant to a synthetic tape."""
    out: list[date] = []
    d = start
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _bar_times(day: date, bars: int) -> list[datetime]:
    """
    Bars spread across the regular session, in tz-naive UTC like the pipeline.

    The last bar is 15:59 ET (20:59 UTC), never 16:00. Minute bars are stamped at
    minute start, so no bar occupies the expiration instant -- and a fixture that
    puts one there hides the whole class of bug where settlement is driven off a
    bar timestamp instead of a session boundary.
    """
    open_utc = datetime(day.year, day.month, day.day, 14, 30)
    if bars == 1:
        return [open_utc]
    # 14:30 through 20:59 inclusive is 390 minute-bars; span the 389 gaps.
    step = 389 / (bars - 1)
    return [open_utc + timedelta(minutes=round(i * step)) for i in range(bars)]


def build_day_frames(spec: LakeSpec, day_index: int, day: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Option and stock frames for one day, in canonical schema order."""
    first_day = spec.start
    spot = spec.price_on(day_index)
    expirations = [
        datetime.combine(first_day + timedelta(days=offset), datetime.min.time()).replace(hour=21)
        for offset in spec.expiry_offsets
    ]

    option_rows: list[dict] = []
    stock_rows: list[dict] = []

    for ts in _bar_times(day, spec.bars_per_day):
        stock_rows.append({
            "timestamp": ts,
            "underlying_open": spot, "underlying_high": spot * 1.002,
            "underlying_low": spot * 0.998, "underlying_close": spot,
            "underlying_vwap": spot, "underlying_volume": 1_000_000,
            "underlying_trade_count": 5_000,
        })

        for expiration in expirations:
            years = max(1e-9, (expiration - ts).total_seconds() / SECONDS_PER_YEAR)
            if years <= 0:
                continue
            for strike in spec.strikes:
                for flag in ("c", "p"):
                    symbol = occ_symbol(spec.ticker, expiration.date(), flag, strike)
                    g = black_scholes(spot, strike, years, spec.risk_free_rate,
                                      spec.implied_vol, flag == "c")
                    price = round(max(0.01, g["price"]), 2)

                    stale = symbol in spec.stale_symbols
                    fallback = symbol in spec.fallback_iv_symbols
                    unpriced = symbol in spec.unpriced_adjusted_symbols

                    option_rows.append({
                        "symbol": symbol,
                        "timestamp": ts,
                        "stock_timestamp": ts,
                        "open": price, "high": round(price * 1.01, 2),
                        "low": round(price * 0.99, 2), "close": price, "vwap": price,
                        "valuation_price": price,
                        "valuation_price_source": "vwap",
                        "iv_input_type": "trade_vwap",
                        "volume": 0 if stale else spec.volume_per_bar,
                        "trade_count": 0 if stale else 50,
                        "underlying_price": spot,
                        "pricing_underlying_price": spot,
                        "strike": strike,
                        "pricing_strike": strike,
                        "pricing_valuation_price": price,
                        "flag": flag,
                        "expiration": expiration,
                        "T": years,
                        "r": spec.risk_free_rate,
                        "rate_source": "fred_treasury_cmt",
                        "q": 0.0,
                        "contract_multiplier": 100.0,
                        "quote_multiplier": 100.0,
                        "deliverable_equity_amount": 100.0,
                        "deliverable_cash_amount": 0.0,
                        "deliverable_count": 1,
                        "deliverable_status": "known_from_contract_master",
                        "adjusted_root": None,
                        "standard_underlying": None,
                        "adjusted_pricing_status": (
                            "unpriced_adjusted_contract" if unpriced else "standard_contract"
                        ),
                        "is_adjusted_contract": unpriced,
                        "is_stale": stale,
                        "dt_stock_min": 0.0,
                        "iv_failed": False,
                        "iv_status": "ok",
                        "iv_is_model_fallback": fallback,
                        "pricing_model": "leisen_reimer_american_discrete_dividend",
                        "model_validation_status": "internal_bsm_sanity_not_vendor_validated",
                        "smoothed_iv": spec.implied_vol,
                        # The pipeline nulls model columns for fallback rows so
                        # they cannot look tradable.
                        "theoretical_price": None if fallback else price,
                        "theoretical_value": None if fallback else price * 100.0,
                        "delta": None if fallback else g["delta"],
                        "gamma": None if fallback else g["gamma"],
                        "theta": None if fallback else g["theta"],
                        "vega": None if fallback else g["vega"],
                        "rho": None if fallback else g["rho"],
                    })

    options = pl.DataFrame(option_rows).select(OPTIONS_ENRICHED_SCHEMA)
    stock = pl.DataFrame(stock_rows).select(STOCK_SCHEMA)
    return options, stock


def write_lake(root: Path, spec: LakeSpec) -> Path:
    """
    Materialize the lake and return the data root.

    Uses the same zstd settings and ``_SUCCESS`` manifest as the pipeline, so a
    reader that depends on either behaves identically here.
    """
    ticker_dir = Path(root) / spec.ticker
    days = _trading_days(spec.start, spec.trading_days)

    for i, day in enumerate(days):
        options, stock = build_day_frames(spec, i, day)
        if options.is_empty():
            continue
        day_dir = ticker_dir / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
        day_dir.mkdir(parents=True, exist_ok=True)

        options.write_parquet(day_dir / "options_enriched.parquet",
                              compression="zstd", compression_level=3, statistics=True)
        stock.write_parquet(day_dir / "stock.parquet",
                            compression="zstd", compression_level=3, statistics=True)

        (day_dir / "_SUCCESS").write_text(json.dumps({
            "status": "COMPLETE",
            "pipeline_version": "2.0.0",
            "row_counts": {
                "options_enriched.parquet": options.height,
                "stock.parquet": stock.height,
            },
        }, indent=2, sort_keys=True))

    return Path(root)


def flat_lake(root: Path, **kw) -> Path:
    """Flat tape at 100, the simplest case for exact ledger assertions."""
    return write_lake(root, LakeSpec(underlying_path=lambda i: 100.0, **kw))


def ramp_lake(root: Path, *, per_day: float = 1.0, **kw) -> Path:
    """Steadily rising tape, which drives short calls into the money."""
    return write_lake(root, LakeSpec(underlying_path=lambda i: 100.0 + per_day * i, **kw))


def crash_lake(root: Path, *, per_day: float = 2.0, **kw) -> Path:
    """Steadily falling tape, which tests long-leg losses and put assignment."""
    return write_lake(root, LakeSpec(underlying_path=lambda i: 100.0 - per_day * i, **kw))
