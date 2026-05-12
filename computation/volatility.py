"""
Volatility estimation and market regime detection.

Implements EWMA volatility with regime-based adjustments,
mirroring the logic from main.py but as reusable functions.
"""

import numpy as np
import pandas as pd



DEFAULT_REGIME_MAP = {
    "crash": 1.8,
    "high_vol": 1.3,
    "normal": 1.0,
    "rally": 0.9,
}


def compute_log_returns(closes: pd.Series) -> pd.Series:
    return np.log(closes / closes.shift(1))


def compute_ewma_volatility(
    log_returns: pd.Series,
    lambda_: float = 0.94,
    window: int = 60,
) -> np.ndarray:
    var = log_returns.iloc[:window].var()
    ewma_var = []

    for r in log_returns:
        if pd.isna(r):
            ewma_var.append(var)
            continue
        var = lambda_ * var + (1 - lambda_) * (r ** 2)
        ewma_var.append(var)

    return np.sqrt(ewma_var) * np.sqrt(252)


def compute_z_scores(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of a series."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def detect_regime(return_z: float, volume_z: float) -> str:
    if return_z < -2:
        return "crash"
    elif return_z > 2:
        return "rally"
    elif volume_z > 1.5:
        return "high_vol"
    else:
        return "normal"


def apply_regime_adjustments(
    volatility: np.ndarray,
    return_z_scores: pd.Series,
    volume_z_scores: pd.Series,
    regime_map: dict[str, float] | None = None,
    window: int = 60,
) -> tuple[np.ndarray, list[str]]:
    if regime_map is None:
        regime_map = DEFAULT_REGIME_MAP

    adjusted = volatility.copy()
    N = len(volatility)
    regimes = ["normal"] * N

    for i in range(window, N):
        regime = detect_regime(
            return_z_scores.iloc[i],
            volume_z_scores.iloc[i],
        )
        regimes[i] = regime
        adjusted[i] = volatility[i] * regime_map[regime]

    return adjusted, regimes


def full_volatility_pipeline(
    data: pd.DataFrame,
    lambda_: float = 0.94,
    window: int = 60,
    regime_map: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    closes = data["Close"]
    volumes = data["Volume"]

    log_returns = compute_log_returns(closes)
    base_vol = compute_ewma_volatility(log_returns, lambda_, window)

    return_z = compute_z_scores(log_returns, window)
    volume_z = compute_z_scores(volumes, window)

    adjusted_vol, regimes = apply_regime_adjustments(
        base_vol, return_z, volume_z, regime_map, window
    )

    return base_vol, adjusted_vol, regimes
