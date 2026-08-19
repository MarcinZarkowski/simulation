import duckdb
from computation.Fetcher.Fetcher import Fetcher

fetcher = Fetcher()
data = fetcher.fetchData("SPY", "2022-06-01", "2024-06-01")
stock_df = data["stock"]
options_df = data["options"]

print("Stock timestamps:", stock_df["timestamp"].head(5).to_list())
print("Options timestamps:", options_df["timestamp"].head(5).to_list())
print("Total stock rows:", len(stock_df))
print("Total options rows:", len(options_df))
