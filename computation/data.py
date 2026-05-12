import yfinance as yf
import pandas as pd


def fetch_history(
    ticker: str,
    period: str = "max",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:

    tk = yf.Ticker(ticker)
    if start and end:
        data = tk.history(start=start, end=end, interval=interval)
    else:
        data = tk.history(period=period, interval=interval)
    return data


def get_price_on_date(data: pd.DataFrame, date: pd.Timestamp) -> float | None:
    date = pd.Timestamp(date).normalize()
    if date in data.index.normalize():
        idx = data.index.normalize().get_loc(date)
        return float(data["Close"].iloc[idx])
    return None


def get_available_dates(data: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(data.index.tolist())
