import streamlit as st
import pandas as pd
import numpy as np
from frontend.tabs.preprocessing import preprocessing_tab
from frontend.tabs.arima_forecast import arima_forecast_tab
from frontend.tabs.arma_forecast import arma_forecast_tab
from frontend.tabs.sarima_forecast import sarima_forecast_tab
from frontend.tabs.garch_forecast import garch_forecast_tab
from frontend.tabs.lstm_forecast import lstm_forecast_tab
from frontend.tabs.transformer_forecast import transformer_forecast_tab
from frontend.tabs.model_comparison import model_comparison_tab
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Streamlit configuration
st.set_page_config(layout="wide", page_title="Crypto Forecasting Dashboard")

# Load custom CSS
with open("frontend/styles/streamlit_style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar setup
st.sidebar.header("Crypto Forecasting Settings")
st.sidebar.image("frontend/images/Pic1.jpg", use_container_width=True)

# Shared inputs
crypto_options = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
    "SOL-USD", "DOT-USD", "DOGE-USD", "MATIC-USD", "LTC-USD"
]
crypto_symbol = st.sidebar.selectbox("Cryptocurrency Symbol", crypto_options, index=0)
target_column = st.sidebar.selectbox("Target Column", ["Open", "High", "Low", "Close", "Volume"], index=3)
prediction_ahead = st.sidebar.number_input("Prediction Days Ahead", min_value=1, max_value=30, value=20, step=1)
interval = st.sidebar.selectbox("Interval", ["1h", "4h", "1d"], index=2)
lookback_days = st.sidebar.slider("Lookback Period (Days)", min_value=30, max_value=365, value=200)
period = f"{lookback_days}d"

# Main page header
st.title("Crypto Forecasting Dashboard")

# Initialize session state for storing data and model results
if 'btc_data' not in st.session_state:
    st.session_state['btc_data'] = None
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = {
        'ARIMA': {'forecast': None, 'metrics': None, 'test_index': None, 'future_index': None},
        'ARMA': {'forecast': None, 'metrics': None, 'test_index': None, 'future_index': None},
        'SARIMA': {'forecast': None, 'metrics': None, 'test_index': None, 'future_index': None},
        'GARCH': {'forecast': None, 'metrics': None, 'test_index': None, 'future_index': None},
        'LSTM': {'forecast': None, 'metrics': None, 'test_index': None, 'future_index': None},
        'Transformer': {'forecast': None, 'metrics': None, 'test_index': None, 'future_index': None}
    }

# # Tabs for different models, preprocessing, and comparison
# tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
#     "Preprocessing", "ARIMA Forecast", "ARMA Forecast", "SARIMA Forecast", 
#     "GARCH Forecast", "LSTM Forecast", "Transformer Forecast", "Model Comparison"
# ])

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Preprocessing", "ARIMA Forecast", "ARMA Forecast", "SARIMA Forecast", 
    "GARCH Forecast", "LSTM Forecast", "Model Comparison"
])

# Call the respective tab functions
with tab0:
    preprocessing_tab(crypto_symbol, period, interval, target_column)

with tab1:
    arima_forecast_tab(crypto_symbol, period, interval, target_column, prediction_ahead)

with tab2:
    arma_forecast_tab(crypto_symbol, period, interval, target_column, prediction_ahead)

with tab3:
    sarima_forecast_tab(crypto_symbol, period, interval, target_column, prediction_ahead)

with tab4:
    garch_forecast_tab(crypto_symbol, period, interval, target_column, prediction_ahead)

with tab5:
    lstm_forecast_tab(crypto_symbol, period, interval, target_column, prediction_ahead)

# with tab6:
#     transformer_forecast_tab(crypto_symbol, period, interval, target_column, prediction_ahead)

with tab6:
    model_comparison_tab(crypto_symbol, period, interval, target_column, prediction_ahead)