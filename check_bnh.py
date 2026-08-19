from computation.Fetcher.Fetcher import Fetcher
import pandas as pd

f = Fetcher()
data = f.fetchData("SPY", "2022-06-01", "2024-06-01")
df = data["stock"].to_pandas()
if len(df) > 0:
    first = df.iloc[0]["Close"]
    last = df.iloc[-1]["Close"]
    print(f"First timestamp: {df.iloc[0]['timestamp']}, Close: {first}")
    print(f"Last timestamp: {df.iloc[-1]['timestamp']}, Close: {last}")
    print(f"Return: {(last / first - 1) * 100:.2f}%")
else:
    print("No data.")
