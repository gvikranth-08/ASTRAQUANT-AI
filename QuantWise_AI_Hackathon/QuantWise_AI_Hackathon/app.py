import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantwise.data import load_market_data, make_demo_data
from quantwise.features import add_features, FEATURE_COLUMNS
from quantwise.model import train_and_predict
from quantwise.backtest import run_backtest, performance_metrics

st.set_page_config(
    page_title="QuantWise AI | Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #080D18; color: #E5EAF3; }
[data-testid="stSidebar"] { background: #101827; border-right: 1px solid #243149; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
.hero { padding: 24px 28px; border: 1px solid #263650; border-left: 5px solid #F97316;
        border-radius: 16px; background: linear-gradient(110deg,#172554,#111827 65%,#101827); margin-bottom: 18px; }
.hero h1 { color: #F8FAFC; margin: 0; font-size: 30px; letter-spacing: 1px; }
.hero p { color: #A8B6CC; margin: 7px 0 0 0; }
.kpi { background: linear-gradient(145deg,#151F31,#0F1725); border: 1px solid #27364E;
       border-radius: 14px; padding: 16px 18px; min-height: 112px; }
.kpi-label { color: #91A2BA; font-size: 11px; letter-spacing: 1.1px; text-transform: uppercase; }
.kpi-value { color: #F8FAFC; font-size: 25px; font-weight: 750; margin-top: 8px; }
.kpi-sub { color: #8293AC; font-size: 11px; margin-top: 4px; }
.buy { color: #34D399 !important; } .sell { color: #FB7185 !important; } .hold { color: #FBBF24 !important; }
.section-note { color: #91A2BA; font-size: 13px; }
div[data-testid="stMetric"] { background:#121C2C; border:1px solid #27364E; padding:14px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

def kpi(label, value, sub="", color=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {color}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True
    )

def make_price_chart(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.06,
                        row_heights=[.72, .28])
    fig.add_trace(go.Candlestick(
        x=data.index, open=data.Open, high=data.High, low=data.Low, close=data.Close,
        name="Price", increasing_line_color="#22C55E", decreasing_line_color="#F43F5E"
    ), row=1, col=1)
    for col, color in [("EMA_9", "#F59E0B"), ("EMA_21", "#60A5FA")]:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col, line=dict(color=color, width=1.5)), row=1, col=1)
    colors = np.where(data.Close >= data.Open, "#22C55E", "#F43F5E")
    fig.add_trace(go.Bar(x=data.index, y=data.Volume, name="Volume", marker_color=colors, opacity=.75), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=540, margin=dict(l=10,r=10,t=30,b=10),
                      paper_bgcolor="#080D18", plot_bgcolor="#0D1523",
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.03),
                      hovermode="x unified")
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="#1E2A3D")
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="#1E2A3D")
    return fig

def equity_chart(bt):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=(bt.Cum_Strategy-1)*100, name="QuantWise strategy",
                             line=dict(color="#34D399", width=2.5), fill="tozeroy",
                             fillcolor="rgba(52,211,153,.08)"))
    fig.add_trace(go.Scatter(x=bt.index, y=(bt.Cum_Market-1)*100, name="Buy & Hold",
                             line=dict(color="#60A5FA", width=2)))
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10,r=10,t=25,b=10),
                      paper_bgcolor="#080D18", plot_bgcolor="#0D1523",
                      yaxis_title="Cumulative return (%)", hovermode="x unified")
    fig.update_xaxes(gridcolor="#1E2A3D"); fig.update_yaxes(gridcolor="#1E2A3D")
    return fig

with st.sidebar:
    st.markdown("## ⚙️ QUANTWISE CONTROLS")
    ticker = st.text_input("Stock ticker", "RELIANCE.NS", help="Examples: TCS.NS, INFY.NS, HDFCBANK.NS, AAPL")
    period = st.selectbox("Intraday lookback", ["5d", "15d", "30d"], index=1)
    st.markdown("---")
    st.markdown("### Risk controls")
    confidence = st.slider("Minimum model confidence", 0.35, 0.90, 0.55, 0.01)
    transaction_cost = st.number_input("Transaction cost + slippage (%)", 0.0, 0.50, 0.03, 0.01)
    st.markdown("---")
    demo_mode = st.toggle("Use synthetic demo data", value=False,
                          help="Use generated sample prices if market data is unavailable.")
    st.caption("Data: Yahoo Finance via yfinance · 5-minute candles")
    run = st.button("↻  RUN ANALYSIS", type="primary", use_container_width=True)

st.markdown("""
<div class="hero">
  <h1>QUANTWISE AI <span style="color:#FB923C">// MARKET INTELLIGENCE</span></h1>
  <p>Machine learning signals · Mathematical indicators · Risk-aware backtesting</p>
</div>
""", unsafe_allow_html=True)

if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = True
if run:
    st.session_state.analysis_run = True

if st.session_state.analysis_run:
    with st.spinner("Loading market data and training the signal engine..."):
        try:
            if demo_mode:
                raw = make_demo_data()
                source_label = "SYNTHETIC DEMO DATA"
            else:
                raw = load_market_data(ticker, period)
                source_label = "YAHOO FINANCE · 5-MINUTE DATA"
                if raw.empty:
                    if st.button("Load synthetic demo data instead"):
                        raw = make_demo_data()
                        source_label = "SYNTHETIC DEMO DATA"
            if raw.empty:
                st.error("No market data returned. Check the ticker, internet connection, or try demo mode.")
                st.stop()
            data = add_features(raw)
            if len(data) < 120:
                st.error(f"Only {len(data)} usable candles found. Choose a longer lookback or use demo mode.")
                st.stop()
            model, test, accuracy, importance = train_and_predict(data, confidence)
            bt = run_backtest(test, transaction_cost)
            metrics = performance_metrics(bt)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    latest = test.iloc[-1]
    signal = int(latest["Filtered_Signal"])
    signal_text = {1:"BUY", -1:"SELL", 0:"HOLD"}[signal]
    signal_class = {1:"buy", -1:"sell", 0:"hold"}[signal]
    confidence_pct = latest["Confidence"] * 100
    price = float(latest["Close"])
    change = float(latest["Return_1"] * 100)

    st.caption(f"● {source_label}  |  {ticker.upper()}  |  Last candle: {test.index[-1]}")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi("Last traded price", f"₹{price:,.2f}", f"Latest candle return: {change:+.2f}%")
    with c2: kpi("AI signal", signal_text, "Confidence-filtered prediction", signal_class)
    with c3: kpi("Model confidence", f"{confidence_pct:.1f}%", f"Minimum required: {confidence:.0%}")
    with c4: kpi("Test accuracy", f"{accuracy*100:.1f}%", "Out-of-sample classification")
    with c5:
        ret = metrics["strategy_return"]
        kpi("Strategy return", f"{ret:+.2f}%", f"After {transaction_cost:.2f}% cost per position change",
            "buy" if ret >= 0 else "sell")

    st.markdown("### 📊 Market overview")
    st.plotly_chart(make_price_chart(data.tail(250)), use_container_width=True)

    left, right = st.columns([1.45,1])
    with left:
        st.markdown("### 🧪 Strategy backtest")
        st.plotly_chart(equity_chart(bt), use_container_width=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Buy & Hold", f'{metrics["market_return"]:+.2f}%')
        m2.metric("Max drawdown", f'{metrics["max_drawdown"]:.2f}%')
        m3.metric("Sharpe ratio", f'{metrics["sharpe"]:.2f}')
    with right:
        st.markdown("### 🧠 What drives the model?")
        imp = importance.sort_values("Importance", ascending=True)
        fig = go.Figure(go.Bar(x=imp.Importance, y=imp.Feature, orientation="h",
                               marker_color="#818CF8"))
        fig.update_layout(template="plotly_dark", height=310, margin=dict(l=5,r=5,t=5,b=5),
                          paper_bgcolor="#080D18", plot_bgcolor="#0D1523",
                          xaxis_title="Relative importance")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Technical snapshot")
        st.metric("RSI (14)", f'{latest["RSI_14"]:.1f}')
        st.metric("MACD histogram", f'{latest["MACD_Hist"]:.4f}')
        st.metric("Bollinger position", f'{latest["BB_Position"]:.2f}')

    st.markdown("### 📐 Mathematical engine")
    with st.expander("View formulas used in this project"):
        st.markdown(r"""
        - **EMA:** \(EMA_t = \alpha P_t + (1-\alpha)EMA_{t-1}\), where \(\alpha=2/(n+1)\).
        - **RSI:** \(RSI=100-\frac{100}{1+RS}\), where \(RS=AvgGain/AvgLoss\).
        - **MACD:** \(MACD=EMA_{12}-EMA_{26}\); signal line is the 9-period EMA of MACD.
        - **Bollinger Bands:** \(Upper/Lower=SMA_{20}\pm2\sigma_{20}\).
        - **Volatility:** rolling standard deviation of candle returns.
        - **Sharpe ratio:** annualized mean strategy return divided by annualized return volatility.
        - **Maximum drawdown:** minimum of \(Equity_t / RunningPeak_t - 1\).
        """)
    st.markdown("### 📋 Recent model predictions")
    display_cols = ["Close","RSI_14","Raw_Signal","Filtered_Signal","Confidence","Strategy_Ret"]
    view = test[display_cols].tail(30).copy()
    view["Raw_Signal"] = view["Raw_Signal"].map({1:"BUY",-1:"SELL",0:"HOLD"})
    view["Filtered_Signal"] = view["Filtered_Signal"].map({1:"BUY",-1:"SELL",0:"HOLD"})
    st.dataframe(view, use_container_width=True)
    csv = test.to_csv().encode("utf-8")
    st.download_button("⬇ Download predictions as CSV", csv, f"{ticker.replace('.','_')}_quantwise_predictions.csv",
                       "text/csv")
    st.caption("Educational research prototype only. Predictions are uncertain, backtests are not guarantees, and this is not investment advice.")
else:
    st.info("Choose a ticker and click **RUN ANALYSIS** to start.")
