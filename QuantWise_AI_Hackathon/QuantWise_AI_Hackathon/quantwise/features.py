import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "MA_Diff", "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "Return_1", "Return_3", "Volatility_10", "Volume_Change",
    "BB_Position", "ATR_Percent"
]

def add_features(raw: pd.DataFrame, horizon: int = 3, threshold: float = 0.0015) -> pd.DataFrame:
    df = raw.copy()
    close = df["Close"]
    df["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA_21"] = close.ewm(span=21, adjust=False).mean()
    df["MA_Diff"] = (df["EMA_9"] - df["EMA_21"]) / close.replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - 100 / (1 + rs)
    df["RSI_14"] = df["RSI_14"].fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    df["Return_1"] = close.pct_change(1)
    df["Return_3"] = close.pct_change(3)
    df["Volatility_10"] = df["Return_1"].rolling(10).std()
    df["Volume_Change"] = df["Volume"].pct_change().replace([np.inf,-np.inf], np.nan)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    band_width = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    df["BB_Position"] = ((close - df["BB_Lower"]) / band_width).clip(-2, 3)

    prev_close = close.shift(1)
    tr = pd.concat([(df["High"]-df["Low"]), (df["High"]-prev_close).abs(),
                    (df["Low"]-prev_close).abs()], axis=1).max(axis=1)
    df["ATR_Percent"] = tr.rolling(14).mean() / close.replace(0, np.nan)

    future_return = close.shift(-horizon) / close - 1
    df["Target"] = 0
    df.loc[future_return > threshold, "Target"] = 1
    df.loc[future_return < -threshold, "Target"] = -1
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ["Target"])
    # Last horizon observations have no future label and must not enter training/testing.
    if horizon:
        df = df.iloc[:-horizon]
    return df
