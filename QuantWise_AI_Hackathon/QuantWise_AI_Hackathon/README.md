# QuantWise AI — Hackathon Edition

A Python + Streamlit stock-market research dashboard combining technical indicators, a Random Forest classifier, confidence filtering, and a transaction-cost-aware historical simulation.

## Features

- Dark fintech dashboard with candlestick and volume charts
- Yahoo Finance 5-minute OHLCV data (`yfinance`)
- EMA 9/21, RSI 14, MACD, Bollinger position, ATR percentage, returns and volatility
- Random Forest classification: BUY (+1), HOLD (0), SELL (-1)
- Confidence threshold to suppress low-confidence predictions
- Chronological train/test split with a small purge gap
- Backtest with transaction costs, Buy & Hold comparison, Sharpe ratio and max drawdown
- Feature-importance chart and downloadable CSV
- Synthetic demo mode for presentations when internet/API data is unavailable

## Quick start

### 1. Create and activate a virtual environment (recommended)

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

Open the local URL shown in your terminal (usually `http://localhost:8501`).

## Demo walkthrough for judges

1. Start with `RELIANCE.NS` or another Yahoo Finance ticker such as `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`.
2. Choose the lookback period and confidence threshold.
3. Click **RUN ANALYSIS**.
4. Explain how the technical indicators become model features.
5. Show the BUY/SELL/HOLD prediction and its confidence.
6. Open the mathematical engine to explain RSI, EMA, MACD, Bollinger Bands and Sharpe ratio.
7. Compare the strategy equity curve with Buy & Hold.
8. Show feature importance and download the CSV.
9. If live data is unavailable, enable **Use synthetic demo data** and clearly disclose that it is simulated data.

## Model and mathematical design

### Labels

For a 3-candle (15-minute) horizon:

`future_return = Close[t+3] / Close[t] - 1`

- BUY (+1) if future return is greater than +0.15%
- SELL (-1) if future return is less than -0.15%
- HOLD (0) otherwise

### Machine learning

RandomForestClassifier with 250 trees, maximum depth 7, minimum leaf size 5, and balanced subsampling class weights. The data is split chronologically (first 75% for training, last 25% for testing) with a three-row purge gap.

### Important interpretation

The classifier predicts a three-candle direction label. The illustrative backtest applies each candle's filtered signal to the following candle return. This mismatch is a simplification for a hackathon prototype, not a production execution simulator. A more rigorous extension would simulate 15-minute holding periods, overlapping positions, exchange fees, taxes, bid/ask spread, latency, and walk-forward validation.

## Data and limitations

- Market data is fetched from Yahoo Finance using `yfinance`; availability, symbols, and intraday history limits are controlled by that provider.
- For NSE symbols, use the `.NS` suffix (for example `RELIANCE.NS`).
- The project is an educational research prototype, not investment advice or an automated trading system.
- Backtest results are historical and do not guarantee future performance. The reported test accuracy is not a measure of profitability.

## Project structure

```text
QuantWise_AI_Hackathon/
├── app.py
├── requirements.txt
├── README.md
└── quantwise/
    ├── __init__.py
    ├── data.py
    ├── features.py
    ├── model.py
    └── backtest.py
```
