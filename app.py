import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Multi-Oscillator Confluence Screener")

# =========================
# SETTINGS
# =========================
mode = st.radio("Scan Mode", ["Custom List", "S&P 500", "Russell 2000"])

tickers_input = st.text_area(
    "Tickers",
    "AAPL,MSFT,TSLA,NVDA,AMZN,META,GOOGL"
)

threshold = st.slider("Confluence Threshold (%)", 10, 100, 80)

timeframe = st.radio(
    "Timeframe",
    ["1H", "4H", "1D", "1W"],
    index=2,
    horizontal=True,
)

# Map UI label → (yfinance interval, download period, resample to 4H?)
# yfinance has no native 4H interval, so we download 1H and resample
TIMEFRAME_MAP = {
    "1H": ("1h",  "60d", False),
    "4H": ("1h",  "60d", True),   # resample=True → aggregate to 4H bars
    "1D": ("1d",  "6mo", False),
    "1W": ("1wk", "5y",  False),
}

# =========================
# LOAD SYMBOLS
# =========================
@st.cache_data
def load_sp500():
    try:
        df = pd.read_csv(
            "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        )
        return df["Symbol"].tolist()
    except:
        return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA"]

@st.cache_data
def load_russell2000():
    try:
        url = (
            "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
            "/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
        )
        df = pd.read_csv(url, skiprows=9)
        df.columns = df.columns.str.strip()
        tickers = df["Ticker"].dropna().astype(str).str.strip()
        tickers = tickers[tickers.str.match(r"^[A-Z]{1,5}$")]
        return tickers.tolist()
    except:
        st.warning("Could not fetch Russell 2000 holdings — falling back to a small sample.")
        return ["SMCI","CROX","BOOT","LUMN","PRCT","ACVA","AMSF","ESTE","HIMS","SKYW"]

if mode == "S&P 500":
    symbols = load_sp500()
elif mode == "Russell 2000":
    symbols = load_russell2000()
else:
    symbols = [s.strip().upper() for s in tickers_input.split(",")]

# =========================
# INDICATORS
# =========================
def compute_indicators(df):
    rsi    = ta_rsi(df["Close"], 14)
    mfi    = ta_mfi(df, 14)
    cci    = ta_cci(df, 20)
    stoch  = ta_stoch(df, 14)
    zscore = ta_zscore(df["Close"], 20)
    return rsi, mfi, cci, stoch, zscore

def ta_rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def ta_mfi(df, length):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos = mf.where(tp > tp.shift(1), 0).rolling(length).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(length).sum()
    return 100 - (100 / (1 + pos / neg))

def ta_cci(df, length):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(length).mean()
    mad = (tp - sma).abs().rolling(length).mean()
    return (tp - sma) / (0.015 * mad)

def ta_stoch(df, length):
    low  = df["Low"].rolling(length).min()
    high = df["High"].rolling(length).max()
    return 100 * (df["Close"] - low) / (high - low)

def ta_zscore(series, length):
    mean = series.rolling(length).mean()
    std  = series.rolling(length).std()
    return (series - mean) / std

# =========================
# 4H RESAMPLER
# =========================
def resample_4h(df):
    """Aggregate 1H OHLCV bars into 4H bars."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    ohlcv = {
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }
    return df.resample("4h").agg(ohlcv).dropna()

# =========================
# ANALYSIS
# =========================
def analyze(symbol, interval, period, do_resample):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)

        if df.empty:
            return None

        # Flatten MultiIndex columns from newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if do_resample:
            df = resample_4h(df)

        if len(df) < 50:
            return None

        rsi, mfi, cci, stoch, zscore = compute_indicators(df)

        last = -1

        rsi_ob   = float(rsi.iloc[last])   >= 70
        rsi_os   = float(rsi.iloc[last])   <= 30
        mfi_ob   = float(mfi.iloc[last])   >= 80
        mfi_os   = float(mfi.iloc[last])   <= 20
        cci_ob   = float(cci.iloc[last])   >= 100
        cci_os   = float(cci.iloc[last])   <= -100
        stoch_ob = float(stoch.iloc[last]) >= 80
        stoch_os = float(stoch.iloc[last]) <= 20
        z_ob     = float(zscore.iloc[last]) >= 2
        z_os     = float(zscore.iloc[last]) <= -2

        count_ob = sum([rsi_ob, mfi_ob, cci_ob, stoch_ob, z_ob])
        count_os = sum([rsi_os, mfi_os, cci_os, stoch_os, z_os])

        score = (count_ob - count_os) / 5 * 100

        return {
            "Ticker":     symbol,
            "Score":      round(float(score), 1),
            "OB Count":   count_ob,
            "OS Count":   count_os,
            "Overbought": score >= threshold,
            "Oversold":   score <= -threshold,
        }

    except:
        return None

# =========================
# RUN
# =========================
if st.button("Run Screener"):

    interval, period, do_resample = TIMEFRAME_MAP[timeframe]
    st.info(f"Scanning **{len(symbols)}** symbols on **{timeframe}** timeframe…")

    results  = []
    progress = st.progress(0)

    for i, sym in enumerate(symbols):
        data = analyze(sym, interval, period, do_resample)
        if data:
            results.append(data)
        progress.progress((i + 1) / len(symbols))

    df = pd.DataFrame(results)

    if df.empty:
        st.warning("No results — try a different timeframe or ticker list.")
    else:
        df = df.sort_values("Score", ascending=False)

        st.subheader("🔥 Strong Overbought")
        st.dataframe(df[df["Overbought"] == True], use_container_width=True)

        st.subheader("🟢 Strong Oversold")
        st.dataframe(df[df["Oversold"] == True], use_container_width=True)

        st.subheader("📊 All Results")
        st.dataframe(df, use_container_width=True)
