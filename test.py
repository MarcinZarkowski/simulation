from computation.Fetcher.Fetcher import Fetcher



def main():
    fetcher = Fetcher()
    data = fetcher.fetchData("SPY", "2024-01-01", "2024-12-31")
    print(data["stock"].head())
    print(data["options"].head())
    from computation.Backtester.StrategyBuilder import StrategyBuilder
    import numpy as np

    time_step_days = fetcher.get_time_step_days(data["stock"])
    print(f"Detected time step: {time_step_days:.6f} days")

    builder = StrategyBuilder()
    
    # Sell a straddle when IV is above 20%
    c1 = builder.cond.iv_above(0.20)
    c2 = builder.cond.every_n_days(7)
    entry_root = builder.cond.and_(c1, c2)
    
    e1 = builder.cond.profit_pct(0.50)  # Close at 50% max profit
    e2 = builder.cond.loss_pct(0.50)    # Or 50% max loss
    exit_root = builder.cond.or_(e1, e2)
    
    legs = [
        {"action": "sell", "type": "CALL", "offset": 0, "dte": 30.0, "contracts": 1},
        {"action": "sell", "type": "PUT", "offset": 0, "dte": 30.0, "contracts": 1}
    ]
    
    builder.add_group(legs, entry_root, exit_root, close_mode="ALL_TOGETHER")
    
    # generate dummy vol since fetcher doesn't fetch it here
    vol = np.full(len(data["stock"]), 0.25)
    
    results = builder.run(data["stock"], options_df=data["options"], adjusted_vol=vol, time_step_days=time_step_days)
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Total Trades: {results['total_trades']}")

if __name__ == "__main__":
    main()
    