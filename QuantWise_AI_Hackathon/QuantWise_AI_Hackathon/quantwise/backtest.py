import numpy as np
import pandas as pd

def run_backtest(test: pd.DataFrame, transaction_cost_pct: float) -> pd.DataFrame:
    bt = test.copy()
    # Signal at candle t is applied to the following candle's close-to-close return.
    bt["Next_Ret"] = bt["Close"].pct_change().shift(-1).fillna(0)
    bt["Position"] = bt["Filtered_Signal"].astype(float)
    turnover = bt["Position"].diff().abs().fillna(bt["Position"].abs())
    cost = transaction_cost_pct / 100.0
    bt["Strategy_Ret"] = bt["Position"] * bt["Next_Ret"] - turnover * cost
    bt["Cum_Strategy"] = (1 + bt["Strategy_Ret"]).cumprod()
    bt["Cum_Market"] = (1 + bt["Next_Ret"]).cumprod()
    return bt

def performance_metrics(bt: pd.DataFrame) -> dict:
    strategy = bt["Strategy_Ret"].replace([np.inf,-np.inf], np.nan).dropna()
    market = bt["Next_Ret"].replace([np.inf,-np.inf], np.nan).dropna()
    equity = bt["Cum_Strategy"]
    peak = equity.cummax()
    drawdown = equity / peak - 1
    # 5-minute bars: 78 regular-session bars/day × 252 trading days/year.
    periods_per_year = 78 * 252
    vol = strategy.std(ddof=1)
    sharpe = (strategy.mean() / vol) * np.sqrt(periods_per_year) if vol and np.isfinite(vol) else 0.0
    active = bt.loc[bt["Position"] != 0, "Strategy_Ret"]
    wins = (active > 0).sum()
    return {
        "strategy_return": (equity.iloc[-1] - 1) * 100 if len(equity) else 0.0,
        "market_return": (bt["Cum_Market"].iloc[-1] - 1) * 100 if len(bt) else 0.0,
        "max_drawdown": drawdown.min() * 100 if len(drawdown) else 0.0,
        "sharpe": float(sharpe),
        "win_rate": (wins / len(active) * 100) if len(active) else 0.0,
    }
