import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import get_yfinance_data

def model_comparison_tab(crypto_symbol, period, interval, target_column, prediction_ahead):
    st.header("Model Comparison and Ensemble Forecast")
    if st.button("Compare Models"):
        with st.spinner("Generating model comparison..."):
            btc_data = get_yfinance_data(crypto_symbol, period, interval)
            if btc_data is not None:
                metrics_data = []
                for model_name, results in st.session_state['model_results'].items():
                    if results['metrics'] is not None:
                        metrics_data.append({
                            'Model': model_name,
                            'MSE': results['metrics']['mse'],
                            'MAE': results['metrics']['mae'],
                            'RMSE': results['metrics']['rmse']
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    st.subheader("Model Performance Metrics")
                    st.dataframe(metrics_df.style.format({"MSE": "{:.2f}", "MAE": "{:.2f}", "RMSE": "{:.2f}"}))

                    plt.figure(figsize=(14, 7))
                    plt.plot(btc_data.index, btc_data[target_column], label='Actual', color='blue', alpha=0.6)

                    forecasts = []
                    for model_name in ['ARIMA', 'LSTM']:
                        results = st.session_state['model_results'][model_name]
                        if results['forecast'] is not None:
                            test_index = results['test_index']
                            future_index = results['future_index']
                            forecast = results['forecast']
                            full_index = list(test_index) + list(future_index)
                            plt.plot(full_index[:len(forecast)], forecast, label=f'{model_name} Forecast', linestyle='--')
                            forecasts.append(forecast)

                    if len(forecasts) > 1:
                        ensemble_forecast = np.mean(forecasts, axis=0)
                        full_index = list(st.session_state['model_results']['ARIMA']['test_index']) + list(st.session_state['model_results']['ARIMA']['future_index'])
                        plt.plot(full_index[:len(ensemble_forecast)], ensemble_forecast, label='Ensemble Forecast', color='black', linewidth=2)

                    plt.axvline(x=btc_data.index[int(len(btc_data) * 0.8)], color='gray', linestyle='--', label='Train/Test Split')
                    plt.title(f'{crypto_symbol} Model Comparison ({target_column})')
                    plt.xlabel('Date')
                    plt.ylabel(target_column)
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)

                    if len(forecasts) > 1:
                        latest_price = float(btc_data[target_column].iloc[-1])
                        last_predicted_price = float(ensemble_forecast[-1])
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown(
                                f"""
                                <div style="display: flex; justify-content: space-around;">
                                    <div class="stCard greenCard">
                                        <h3>Latest {target_column}</h3>
                                        <p style="font-size: 20px;">${latest_price:,.2f}</p>
                                    </div>
                                    <div class="stCard greenCard">
                                        <h3>Ensemble {target_column} After {prediction_ahead} Days</h3>
                                        <p style="font-size: 20px;">${last_predicted_price:,.2f}</p>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    if metrics_data:
                        csv = metrics_df.to_csv(index=False)
                        st.download_button("Download Metrics", csv, "model_metrics.csv", "text/csv")
                else:
                    st.warning("Run at least one model to compare results.")
