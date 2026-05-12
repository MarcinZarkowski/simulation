"""
Simulation Configuration.

This file holds all the hardcoded configuration parameters for the backtester.
"""

VOL_WINDOW = 60
EWMA_LAMBDA = 0.94

# multiplies vol by given multiple depending on the state of the market,
# mimics how IV is often higher in crash/high_vol and lower in rallies
REGIME_MAP = {
    "crash": 1.5,
    "high_vol": 1.2,
    "normal": 1.0,
    "rally": 0.8,
}

RISK_FREE_RATE = 0.06 # 6.00%
DEFAULT_DTE = 30
