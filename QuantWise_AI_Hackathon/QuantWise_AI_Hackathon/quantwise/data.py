import numpy as np
import pandas as pd
import yfinance as yf

def load_market_data(ticker: str, period: str = "5d") -> pd.DataFrame:
    """Download 5-minute OHLCV candles. Yahoo Finance intraday history is limited."""
    df = yf.download(ticker.strip(), period=period, interval="5m", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(df.columns):
        return pd.DataFrame()
    return df[required].dropna().copy()

def make_demo_data(n: int = 900, seed: int = 17) -> pd.DataFrame:
    """Reproducible synthetic intraday-like data for demos when the API is unavailable."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.now().floor("5min"), periods=n, freq="5min")
    drift = 0.00003 + 0.00018 * np.sin(np.arange(n) / 65)
    returns = drift + rng.normal(0, 0.0018, n)
    close = 2400 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.0012, 0.0005, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(5000, 50000, n).astype(float)
    return pd.DataFrame({"Open":open_, "High":high, "Low":low, "Close":close, "Volume":volume}, index=idx)
