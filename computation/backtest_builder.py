"""
Strategy Builder — Python interface for the C++ backtest engine.

Provides ConditionBuilder and StrategyBuilder that serialize strategy
definitions into flat numpy arrays for zero-copy pybind11 transfer.
"""

import numpy as np

# Must match C++ CondType enum
COND_EVERY_N_DAYS   = 0
COND_DELTA_BETWEEN  = 1
COND_IV_ABOVE       = 2
COND_IV_BELOW       = 3
COND_PRICE_ABOVE    = 4
COND_PRICE_BELOW    = 5
COND_DAY_OF_WEEK    = 6
COND_HOLD_TO_EXPIRY = 10
COND_PROFIT_PCT     = 11
COND_LOSS_PCT       = 12
COND_PROFIT_DOLLARS = 13
COND_LOSS_DOLLARS   = 14
COND_DTE_REMAINING  = 15
COND_AND            = 20
COND_OR             = 21
COND_NOT            = 22
COND_TRUE           = 23
COND_FALSE          = 24

CLOSE_ALL_TOGETHER = 0
CLOSE_INDIVIDUALLY = 1
OPT_CALL = 0
OPT_PUT  = 1


class ConditionBuilder:
    """Builds an expression tree of conditions as a flat list."""

    def __init__(self):
        self._nodes: list[tuple[int, float, float, int, int]] = []

    def _add(self, ctype: int, p1=0.0, p2=0.0, c1=-1, c2=-1) -> int:
        idx = len(self._nodes)
        self._nodes.append((ctype, float(p1), float(p2), c1, c2))
        return idx

    # Entry conditions
    def every_n_days(self, n: int) -> int:
        return self._add(COND_EVERY_N_DAYS, p1=n)

    def delta_between(self, low: float, high: float) -> int:
        return self._add(COND_DELTA_BETWEEN, p1=low, p2=high)

    def iv_above(self, threshold: float) -> int:
        return self._add(COND_IV_ABOVE, p1=threshold)

    def iv_below(self, threshold: float) -> int:
        return self._add(COND_IV_BELOW, p1=threshold)

    def price_above(self, level: float) -> int:
        return self._add(COND_PRICE_ABOVE, p1=level)

    def price_below(self, level: float) -> int:
        return self._add(COND_PRICE_BELOW, p1=level)

    def on_days(self, days: list[int]) -> int:
        """Only trigger on specific weekdays. days=[0,2,4] = Mon/Wed/Fri."""
        if not days:
            return self.never()
        
        day_conds = [self._add(COND_DAY_OF_WEEK, p1=d) for d in days]
        combined = day_conds[0]
        for i in range(1, len(day_conds)):
            combined = self.or_(combined, day_conds[i])
        return combined

    # Exit conditions
    def hold_to_expiry(self) -> int:
        return self._add(COND_HOLD_TO_EXPIRY)

    def profit_pct(self, pct: float) -> int:
        return self._add(COND_PROFIT_PCT, p1=pct)

    def loss_pct(self, pct: float) -> int:
        return self._add(COND_LOSS_PCT, p1=pct)

    def profit_dollars(self, amount: float) -> int:
        return self._add(COND_PROFIT_DOLLARS, p1=amount)

    def loss_dollars(self, amount: float) -> int:
        return self._add(COND_LOSS_DOLLARS, p1=amount)

    def dte_remaining(self, days: int) -> int:
        return self._add(COND_DTE_REMAINING, p1=days)

    # Combinators
    def and_(self, a: int, b: int) -> int:
        return self._add(COND_AND, c1=a, c2=b)

    def or_(self, a: int, b: int) -> int:
        return self._add(COND_OR, c1=a, c2=b)

    def not_(self, a: int) -> int:
        return self._add(COND_NOT, c1=a)

    def always(self) -> int:
        return self._add(COND_TRUE)

    def never(self) -> int:
        return self._add(COND_FALSE)

    def to_arrays(self):
        if not self._nodes:
            return (
                np.array([COND_TRUE], dtype=np.int32),
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([-1], dtype=np.int32),
                np.array([-1], dtype=np.int32),
            )
        types, p1s, p2s, c1s, c2s = zip(*self._nodes)
        return (
            np.array(types, dtype=np.int32),
            np.array(p1s, dtype=np.float32),
            np.array(p2s, dtype=np.float32),
            np.array(c1s, dtype=np.int32),
            np.array(c2s, dtype=np.int32),
        )


class StrategyBuilder:
    """High-level builder for multi-leg, multi-group strategies."""

    def __init__(self):
        self.cond = ConditionBuilder()
        self._groups: list[dict] = []
        self._legs: list[dict] = []

    def add_group(self, legs: list[dict], entry_condition: int, exit_condition: int, close_mode: str = "ALL_TOGETHER"):
        group_id = len(self._groups)
        cm = CLOSE_ALL_TOGETHER if close_mode == "ALL_TOGETHER" else CLOSE_INDIVIDUALLY
        self._groups.append({"close_mode": cm, "entry_root": entry_condition, "exit_root": exit_condition})

        for leg in legs:
            self._legs.append({
                "group_id": group_id,
                "buy": 1 if leg["action"] == "buy" else 0,
                "opt_type": OPT_CALL if leg["type"] == "CALL" else OPT_PUT,
                "strike_offset": leg.get("offset", 0),
                "dte": leg.get("dte", 30),
                "contracts": leg.get("contracts", 1),
            })

    def _build_arrays(self):
        cond_type, cond_p1, cond_p2, cond_c1, cond_c2 = self.cond.to_arrays()
        return {
            "leg_group_id":      np.array([l["group_id"] for l in self._legs], dtype=np.int32),
            "leg_buy":           np.array([l["buy"] for l in self._legs], dtype=np.int32),
            "leg_opt_type":      np.array([l["opt_type"] for l in self._legs], dtype=np.int32),
            "leg_strike_offset": np.array([l["strike_offset"] for l in self._legs], dtype=np.int32),
            "leg_dte":           np.array([l["dte"] for l in self._legs], dtype=np.int32),
            "leg_contracts":     np.array([l["contracts"] for l in self._legs], dtype=np.int32),
            "group_close_mode":  np.array([g["close_mode"] for g in self._groups], dtype=np.int32),
            "group_entry_root":  np.array([g["entry_root"] for g in self._groups], dtype=np.int32),
            "group_exit_root":   np.array([g["exit_root"] for g in self._groups], dtype=np.int32),
            "cond_type": cond_type, "cond_param1": cond_p1, "cond_param2": cond_p2,
            "cond_child1": cond_c1, "cond_child2": cond_c2,
        }

    def run(self, prices_df, adjusted_vol, ticker="SPY", starting_balance=10000.0, starting_shares=0, starting_average_cost=0.0, risk_free_rate=0.05):
        import backtest_engine

        opens   = prices_df["Open"].to_numpy().astype(np.float32)
        highs   = prices_df["High"].to_numpy().astype(np.float32)
        lows    = prices_df["Low"].to_numpy().astype(np.float32)
        closes  = prices_df["Close"].to_numpy().astype(np.float32)
        volumes = prices_df["Volume"].to_numpy().astype(np.float32)
        vol     = np.asarray(adjusted_vol, dtype=np.float32)

        # Day-of-week from the DatetimeIndex (0=Mon .. 4=Fri)
        day_of_week = prices_df.index.dayofweek.to_numpy().astype(np.int32)
        arrays = self._build_arrays()
        return backtest_engine.run_backtest(
            opens, highs, lows, closes, volumes, vol, day_of_week,
            starting_balance, starting_shares, starting_average_cost, ticker,
            **arrays,
            r=risk_free_rate
        )
