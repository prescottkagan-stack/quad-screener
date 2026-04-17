import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Quad Rotation Screener — Institutional")

# =========================
# SETTINGS
# =========================
mode = st.radio("Scan Mode", ["Custom List", "S&P 500"])

tickers_input = st.text_area(
    "Tickers (for Custom List)",
    "AAPL,MSFT,TSLA,NVDA,AMZN,META,GOOGL"
)

ob_level = st.slider("Overbought", 50, 100, 80)
os_level = st.slider("Oversold", 0, 50, 20)

rotation_window = st.slider("Rotation Window", 1, 10, 4)
armed_window = st.slider("Armed Window", 1, 30, 12)
slope_threshold = st.slider("Slope Threshold", 0.0, 2.0, 0.75)

# =========================
# LOAD SYMBOLS (STABLE)
# =========================
@st.cache_data
def load_sp500():
    try:
        df = pd.read_csv(
            "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        )
        return df["Symbol"].tolist()
    except:
        # Fallback (never breaks)
        return [
            "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA",
            "BRK-B","UNH","XOM","JNJ","JPM","V","PG","AVGO",
            "HD","MA","CVX","LLY","ABBV","PEP","KO","COST",
            "MRK","WMT","BAC","ADBE","CRM","NFLX","AMD"
        ]

# ✅ DEFINE SYMBOLS (THIS FIXES YOUR ERROR)
if mode == "S&P 500":
    symbols = load_sp500()
else:
    symbols = [s.strip().upper() for s in tickers_input.split(",")]

# =========================
# FUNCTIONS
# =========================
def stochastic(df, length):
    low = df['Low'].rolling(length).min()
    high = df['High'].rolling(length).max()
    k = 100 * (df['Close'] - low) / (high - low)
    return k.fillna(50)

def slope(series, lookback=2):
    return (series - series.shift(lookback)) / lookback

def crossover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))

def crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))

def bars_since(cond):
    idx = np.where(cond)[0]
    if len(idx) == 0:
        return 999
    return len(cond) - idx[-1] - 1

# =========================
# ANALYSIS
# =========================
def analyze(symbol):
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)

        if df.empty or len(df) < 100:
            return None

        k1 = stochastic(df, 9)
        k2 = stochastic(df, 14)
        k3 = stochastic(df, 40)
        k4 = stochastic(df, 60)

        d1 = k1.rolling(3).mean()
        d2 = k2.rolling(3).mean()
        d3 = k3.rolling(4).mean()
        d4 = k4.rolling(10).mean()

        last = -1

        # EXTREMES
        os_series = (k1 <= os_level) & (k2 <= os_level) & (k3 <= os_level) & (k4 <= os_level)
        ob_series = (k1 >= ob_level) & (k2 >= ob_level) & (k3 >= ob_level) & (k4 >= ob_level)

        # ARMED
        bull_armed = bars_since(os_series.values) <= armed_window
        bear_armed = bars_since(ob_series.values) <= armed_window

        # ROTATION CROSS
        bull_cross = (
            bars_since(crossover(k1, d1).values) <= rotation_window and
            bars_since(crossover(k2, d2).values) <= rotation_window and
            bars_since(crossover(k3, d3).values) <= rotation_window and
            bars_since(crossover(k4, d4).values) <= rotation_window
        )

        bear_cross = (
            bars_since(crossunder(k1, d1).values) <= rotation_window and
            bars_since(crossunder(k2, d2).values) <= rotation_window and
            bars_since(crossunder(k3, d3).values) <= rotation_window and
            bars_since(crossunder(k4, d4).values) <= rotation_window
        )

        # SLOPE
        bull_slope = all([
            slope(k1).iloc[last] >= slope_threshold,
            slope(k2).iloc[last] >= slope_threshold,
            slope(k3).iloc[last] >= slope_threshold,
            slope(k4).iloc[last] >= slope_threshold
        ])

        bear_slope = all([
            slope(k1).iloc[last] <= -slope_threshold,
            slope(k2).iloc[last] <= -slope_threshold,
            slope(k3).iloc[last] <= -slope_threshold,
            slope(k4).iloc[last] <= -slope_threshold
        ])

        # FINAL SIGNAL
        bull_signal = bull_armed and bull_cross and bull_slope
        bear_signal = bear_armed and bear_cross and bear_slope

        # STRENGTH SCORE
        strength = (
            abs(k1.iloc[last]-50) +
            abs(k2.iloc[last]-50) +
            abs(k3.iloc[last]-50) +
            abs(k4.iloc[last]-50)
        )

        return {
            "Ticker": symbol,
            "K1": round(k1.iloc[last], 1),
            "K2": round(k2.iloc[last], 1),
            "K3": round(k3.iloc[last], 1),
            "K4": round(k4.iloc[last], 1),
            "Bull Signal": bull_signal,
            "Bear Signal": bear_signal,
            "Bull Armed": bull_armed,
            "Bear Armed": bear_armed,
            "Strength": round(strength, 1)
        }

    except:
        return None

# =========================
# RUN SCREENER
# =========================
if st.button("Run Screener"):

    results = []
    progress = st.progress(0)

    for i, sym in enumerate(symbols):
        data = analyze(sym)
        if data:
            results.append(data)

        progress.progress((i + 1) / len(symbols))

    df = pd.DataFrame(results)

    if df.empty:
        st.warning("No results found")
    else:
        df = df.sort_values("Strength", ascending=False)

        st.subheader("🔥 Top Signals")
        st.dataframe(df.head(25), use_container_width=True)

        st.subheader("🟢 Bull Signals")
        st.dataframe(df[df["Bull Signal"] == True], use_container_width=True)

        st.subheader("🔴 Bear Signals")
        st.dataframe(df[df["Bear Signal"] == True], use_container_width=True)

        st.subheader("📊 Full Data")
        st.dataframe(df, use_container_width=True)
