import streamlit as st
from frontend.tabs.preprocessing import preprocessing_tab
from frontend.tabs.arima_forecast import arima_forecast_tab
from frontend.tabs.sarima_forecast import sarima_forecast_tab
from frontend.tabs.var_forecast import var_forecast_tab
from frontend.tabs.varma_forecast import varma_forecast_tab
from frontend.tabs.garch_forecast import garch_forecast_tab
from frontend.tabs.lstm_forecast import lstm_forecast_tab
from frontend.tabs.model_comparison import model_comparison_tab
import os

def main():
    st.set_page_config(page_title="Crypto Forecasting App", layout="wide")
    
    # Load CSS
    css_path = os.path.join("frontend", "styles", "streamlit_style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Add centered title
    st.markdown(
        """
        <h1 style='text-align: center;'>
            Crypto Price Predictor 📈💰
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state
    if 'model_results' not in st.session_state:
        st.session_state['model_results'] = {
            'ARIMA': {'forecast': None, 'future_index': None, 'metrics': None},
            'SARIMA': {'forecast': None, 'future_index': None, 'metrics': None},
            'VAR': {'forecast': None, 'future_index': None, 'metrics': None},
            'VARMA': {'forecast': None, 'future_index': None, 'metrics': None},
            'GARCH': {'forecast': None, 'future_index': None, 'metrics': None},
            'LSTM': {'forecast': None, 'future_index': None, 'metrics': None}
        }
    if 'btc_data' not in st.session_state:
        st.session_state['btc_data'] = None

    # Sidebar
    image_path = os.path.join("frontend", "images", "crypto_logo.jpg")
    if os.path.exists(image_path):
        st.sidebar.image(image_path, use_container_width=True)
    else:
        st.sidebar.warning("Image not found at frontend/images/crypto_logo.png")
    
    st.sidebar.title("Settings")
    
    # File selection
    csv_files = {
        "15m": "data/btc_15m_data_2018_to_2025.csv",
        "1h": "data/btc_1h_data_2018_to_2025.csv",
        "4h": "data/btc_4h_data_2018_to_2025.csv",
        "1d": "data/btc_1d_data_2018_to_2025.csv"
    }
    interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=3)
    file_path = csv_files[interval]

    # Other settings
    target_column = st.sidebar.selectbox("Target Column", ["Open", "High", "Low", "Close", "Volume"], index=3)
    prediction_ahead = st.sidebar.slider("Prediction Days Ahead", min_value=1, max_value=30, value=20)
    lookback_days = st.sidebar.slider("Lookback Period (Days)", min_value=30, max_value=365, value=250)

    # Tabs
    tabs = ["Preprocessing", "ARIMA", "SARIMA", "VAR", "VARMA", "GARCH", "LSTM", "Model Comparison"]
    selected_tab = st.sidebar.radio("Select Tab", tabs)

    # Render selected tab
    try:
        if selected_tab == "Preprocessing":
            preprocessing_tab(file_path, interval, target_column, lookback_days)
        elif selected_tab == "ARIMA":
            arima_forecast_tab(file_path, interval, target_column, prediction_ahead, lookback_days)
        elif selected_tab == "SARIMA":
            sarima_forecast_tab(file_path, interval, target_column, prediction_ahead, lookback_days)
        elif selected_tab == "VAR":
            var_forecast_tab(file_path, interval, target_column, prediction_ahead, lookback_days)
        elif selected_tab == "VARMA":
            varma_forecast_tab(file_path, interval, target_column, prediction_ahead, lookback_days)
        elif selected_tab == "GARCH":
            garch_forecast_tab(file_path, interval, target_column, prediction_ahead, lookback_days)
        elif selected_tab == "LSTM":
            lstm_forecast_tab(file_path, interval, target_column, prediction_ahead, lookback_days)
        elif selected_tab == "Model Comparison":
            model_comparison_tab(file_path, interval, target_column, prediction_ahead, lookback_days)
    except Exception as e:
        st.error(f"Error in {selected_tab} tab: {str(e)}")

if __name__ == "__main__":
    main()