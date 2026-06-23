import io
import warnings
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ARIMA Forecasting for Indian Stocks", layout="wide")

st.title("ARIMA Forecasting for Indian Stocks")
st.write(
    "Upload or enter Indian stock symbols, download the last 5 years of data from Yahoo Finance, "
    "fit an ARIMA model, and forecast values up to June 2027."
)

DEFAULT_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "LT.NS",
    "BHARTIARTL.NS",
    "HINDUNILVR.NS"
]

TARGET_DATE = pd.Timestamp("2027-06-30")

st.sidebar.header("Input Settings")

input_mode = st.sidebar.radio(
    "Choose input method",
    ["Manual symbols", "Upload CSV"],
    index=0
)

manual_symbols = st.sidebar.text_area(
    "Enter Yahoo Finance Indian stock symbols separated by commas",
    value=", ".join(DEFAULT_STOCKS),
    height=150
)

uploaded_csv = st.sidebar.file_uploader(
    "Upload CSV with one column of stock symbols",
    type=["csv"]
)

price_field = st.sidebar.selectbox(
    "Price field",
    ["Close", "Open", "High", "Low"],
    index=0
)

max_stocks = st.sidebar.slider(
    "Maximum number of stocks to process",
    min_value=1,
    max_value=50,
    value=10
)

run_button = st.sidebar.button("Run Forecast")

def clean_symbols_from_text(text):
    symbols = [x.strip().upper() for x in text.split(",") if x.strip()]
    unique_symbols = []
    for symbol in symbols:
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)
    return unique_symbols

def clean_symbols_from_csv(file_obj):
    df = pd.read_csv(file_obj)
    if df.empty:
        return []

    first_col = df.columns[0]
    symbols = (
        df[first_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    unique_symbols = []
    for symbol in symbols:
        if symbol and symbol not in unique_symbols:
            unique_symbols.append(symbol)
    return unique_symbols

def get_symbols():
    if input_mode == "Upload CSV" and uploaded_csv is not None:
        return clean_symbols_from_csv(uploaded_csv)
    return clean_symbols_from_text(manual_symbols)

def download_stock_data(symbol):
    try:
        df = yf.download(
            symbol,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            multi_level_index=False
        )
        return df
    except Exception:
        return pd.DataFrame()

def prepare_monthly_series(symbol, selected_field):
    df = download_stock_data(symbol)

    if df is None or df.empty:
        return None, f"No data found for {symbol}."

    if selected_field not in df.columns:
        return None, f"{selected_field} column is not available for {symbol}."

    series = df[selected_field].dropna()

    if series.empty:
        return None, f"No valid {selected_field} data available for {symbol}."

    series.index = pd.to_datetime(series.index)
    monthly_series = series.resample("M").last().dropna()

    if len(monthly_series) < 24:
        return None, f"Not enough monthly data to model {symbol}."

    monthly_series.name = symbol
    return monthly_series, None

def select_best_arima(series):
    candidate_orders = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 2),
        (2, 1, 2),
        (3, 1, 1),
        (1, 1, 3)
    ]

    best_model = None
    best_order = None
    best_aic = None

    for order in candidate_orders:
        try:
            model = ARIMA(series, order=order)
            fitted = model.fit()
            if best_aic is None or fitted.aic < best_aic:
                best_aic = fitted.aic
                best_model = fitted
                best_order = order
        except Exception:
            continue

    return best_model, best_order

def months_until_target(last_date, target_date):

