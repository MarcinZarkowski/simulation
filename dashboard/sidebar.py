"""
Sidebar configuration panel.

All simulation parameters live here so they're available to every tab.
Returns a config dict and the cached data.
"""

import streamlit as st
import pandas as pd

from computation.data import fetch_history
from computation.volatility import full_volatility_pipeline
from computation import config


def render_sidebar() -> dict:
    """
    Render the sidebar and return a config dict with all user settings
    plus the fetched data and computed volatility.
    """
    st.sidebar.title("Configuration")
    
    st.sidebar.subheader("Data")
    ticker = st.sidebar.text_input("Ticker Symbol", value="NVDA").upper()

    use_date_range = st.sidebar.checkbox("Custom date range", value=False)
    if use_date_range:
        col1, col2 = st.sidebar.columns(2)
        start_date = col1.date_input("Start", value=pd.Timestamp("2015-01-01"))
        end_date = col2.date_input("End", value=pd.Timestamp.today())
        start_str = str(start_date)
        end_str = str(end_date)
        period = None
    else:
        period = st.sidebar.selectbox(
            "Period",
            ["1y", "2y", "5y", "10y", "max"],
            index=4,
        )
        start_str = None
        end_str = None

    # ── Volatility ────────────────────────────────────────────────────────
    vol_window = config.VOL_WINDOW
    ewma_lambda = config.EWMA_LAMBDA
    regime_map = config.REGIME_MAP

    risk_free_rate = config.RISK_FREE_RATE
    default_dte = config.DEFAULT_DTE


    # ── Fetch data ────────────────────────────────────────────────────────
    data = _fetch_cached(ticker, period, start_str, end_str)

    if data is None or data.empty:
        st.sidebar.error(f"No data found for {ticker}")
        return None

    st.sidebar.success(f"**{ticker}** — {len(data)} bars loaded")

    # ── Compute volatility (cached) ─────────────────────────────────────
    base_vol, adjusted_vol, regimes = _compute_vol_cached(
        data, ewma_lambda, vol_window,
        regime_map["crash"], regime_map["high_vol"],
        regime_map["normal"], regime_map["rally"],
    )

    return {
        "ticker": ticker,
        "data": data,
        "base_vol": base_vol,
        "adjusted_vol": adjusted_vol,
        "regimes": regimes,
        "vol_window": vol_window,
        "ewma_lambda": ewma_lambda,
        "regime_map": regime_map,
        "risk_free_rate": risk_free_rate,
        "default_dte": default_dte,
    }


@st.cache_data(show_spinner="Computing volatility…")
def _compute_vol_cached(
    data: pd.DataFrame,
    ewma_lambda: float,
    vol_window: int,
    crash: float,
    high_vol: float,
    normal: float,
    rally: float,
):
    """Cache volatility pipeline — only recomputes when params change."""
    regime_map = {"crash": crash, "high_vol": high_vol, "normal": normal, "rally": rally}
    base_vol, adjusted_vol, regimes = full_volatility_pipeline(
        data, ewma_lambda, vol_window, regime_map
    )
    return base_vol, adjusted_vol, regimes


@st.cache_data(show_spinner="Fetching market data…")
def _fetch_cached(
    ticker: str,
    period: str | None,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Cache yfinance calls so re-renders don't re-download."""
    return fetch_history(ticker, period=period or "max", start=start, end=end)
