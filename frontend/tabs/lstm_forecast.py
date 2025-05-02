import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import get_yfinance_data, calculate_metrics
from backend.predictor import Predictor

def lstm_forecast_tab(crypto_symbol, period, interval, target_column, prediction_ahead):
    st.header("LSTM Forecasting Model")
    predictor = Predictor()
    predictor.load_models()

    if st.button("Predict with LSTM"):
        with st.spinner("Making LSTM predictions..."):
            btc_data = get_yfinance_data(crypto_symbol, period, interval)
            if btc_data is not None:
                train_size = int(len(btc_data) * 0.8)
                time_step = 60

                forecast = predictor.get_predictions(
                    symbol=crypto_symbol,
                    period=period,
                    interval=interval,
                    target_column=target_column,
                    steps=prediction_ahead,
                    model_type="lstm",
                    time_step=time_step
                )

                # Since predictor only gives future forecast, we need test predictions separately
                test_data = btc_data[train_size - time_step:]
                test_forecast = predictor.predict_lstm(test_data, target_column, len(test_data) - time_step, time_step)
                full_forecast = np.concatenate([test_forecast, forecast])

                mse, mae, rmse = calculate_metrics(test_data[target_column].values[time_step:], test_forecast)
                st.subheader("Model Metrics")
                st.write(f"MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

                st.session_state['model_results']['LSTM'] = {
                    'forecast': full_forecast,
                    'metrics': {'mse': mse, 'mae': mae, 'rmse': rmse},
                    'test_index': btc_data.index[train_size:train_size + len(test_forecast)],
                    'future_index': pd.date_range(start=btc_data.index[-1], periods=prediction_ahead + 1, freq='D')[1:]
                }

                last_price = float(btc_data[target_column].iloc[-1])
                last_predicted_price = float(forecast[-1])

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-around;">
                            <div class="stCard greenCard">
                                <h3>Latest {target_column}</h3>
                                <p style="font-size: 20px;">${last_price:,.2f}</p>
                            </div>
                            <div class="stCard greenCard">
                                <h3>{target_column} After {prediction_ahead} Days</h3>
                                <p style="font-size: 20px;">${last_predicted_price:,.2f}</p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                plt.figure(figsize=(14, 5))
                plt.plot(btc_data.index, btc_data[target_column], label='Actual', color='blue')
                plt.axvline(x=btc_data.index[train_size], color='gray', linestyle='--', label='Train/Test Split')
                test_range = btc_data.index[train_size:train_size + len(test_forecast)]
                plt.plot(test_range, test_forecast, label='Test Predictions', color='orange')
                future_index = pd.date_range(start=btc_data.index[-1], periods=prediction_ahead + 1, freq='D')[1:]
                plt.plot(future_index, forecast, label=f'{prediction_ahead}-Day Forecast', color='red')
                plt.title(f'{crypto_symbol} LSTM Model Predictions ({target_column})')
                plt.xlabel('Date')
                plt.ylabel(target_column)
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)