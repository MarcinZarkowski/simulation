from computation.Fetcher.Fetcher import Fetcher
from computation.Backtester.StrategyBuilder import StrategyBuilder

f = Fetcher()
data = f.fetchData("SPY", "2024-02-01", "2024-05-31")

builder = StrategyBuilder()
builder.add_group(
    legs=[{"action": "buy", "type": "CALL", "offset": 0, "dte": 30, "contracts": 1}],
    entry_condition=builder.cond.every_n_days(1),
    exit_condition=builder.cond.profit_pct(0.50),
    close_mode="ALL_TOGETHER"
)

res = builder.run(data["stock"], data["options"])
print("Total Positions Opened:", res.get("total_positions_opened"))
print("Final PnL:", res.get("total_pnl"))
