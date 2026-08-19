from computation.Fetcher.Fetcher import Fetcher
from computation.Backtester.StrategyBuilder import StrategyBuilder

f = Fetcher()
data = f.fetchData("SPY", "2024-02-01", "2024-05-31")
stock, opts = data["stock"], data["options"]

print("=" * 60)
print("TEST 1: Buy calls every 30 days (should open ~3-4 positions)")
b = StrategyBuilder()
b.add_group(
    legs=[{"action": "buy", "type": "CALL", "offset": 0, "dte": 30, "contracts": 1}],
    entry_condition=b.cond.every_n_days(30),
    exit_condition=b.cond.profit_pct(0.50),
    close_mode="ALL_TOGETHER"
)
r = b.run(stock, opts, starting_balance=10000)
print(f"  Positions opened: {r['total_positions_opened']} (expect ~3-4)")
print(f"  Total PnL: ${r['total_pnl']:.2f}")

print()
print("TEST 2: Sell put spread - Iron Condor wing (margin should block over-leverage)")
b2 = StrategyBuilder()
b2.add_group(
    legs=[
        {"action": "sell", "type": "PUT", "offset": -5, "dte": 30, "contracts": 1},
        {"action": "buy",  "type": "PUT", "offset": -10, "dte": 30, "contracts": 1},
    ],
    entry_condition=b2.cond.every_n_days(30),
    exit_condition=b2.cond.profit_pct(0.50),
    close_mode="ALL_TOGETHER"
)
r2 = b2.run(stock, opts, starting_balance=5000)
print(f"  Positions opened: {r2['total_positions_opened']}")
print(f"  Total PnL: ${r2['total_pnl']:.2f}")

print()
print("TEST 3: Buy calls every day - sanity check returns (should NOT be millions of %)")
b3 = StrategyBuilder()
b3.add_group(
    legs=[{"action": "buy", "type": "CALL", "offset": 0, "dte": 30, "contracts": 1}],
    entry_condition=b3.cond.every_n_days(1),
    exit_condition=b3.cond.profit_pct(0.50),
    close_mode="ALL_TOGETHER"
)
r3 = b3.run(stock, opts, starting_balance=10000)
start = 10000; end = r3["daily_values"][-1]
pct = (end / start - 1) * 100
print(f"  Positions opened: {r3['total_positions_opened']}")
print(f"  Return: {pct:.2f}% (expect < 500%)")
