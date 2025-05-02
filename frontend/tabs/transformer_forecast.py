# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from utils.data_utils import get_yfinance_data, calculate_metrics
# from backend.predictor import Predictor

# def transformer_forecast_tab(crypto_symbol, period, interval, target_column, prediction_ahead):
#     st.header("Transformer Forecasting Model")
#     predictor = Predictor()
#     predictor.load_models()

#     if st.button("Predict with Transformer"):
#         with st.spinner("Making Transformer predictions..."):
#             btc_data = get_yfinance_data(crypto_symbol, period, interval)
#             if btc_data is not None:
#                 train_size = int(len(btc_data) * 0.8)
#                 time_step = 60

#                 forecast = predictor.get_predictions(
#                     symbol=crypto_symbol,
#                     period=period,
#                     interval=interval,
#                     target_column=target_column,
#                     steps=prediction_ahead,
#                     model_type="transformer",
#                     time_step=time_step
#                 )

#                 test_data = btc_data[train_size - time_step:]
#                 test_forecast = predictor.predict_transformer(test_data, target_column, len(test_data) - time_step, time_step)
#                 full_forecast = np.concatenate([test_forecast, forecast])

#                 mse, mae, rmse = calculate_metrics(test_data[target_column].values[time_step:], test_forecast)
#                 st.subheader("Model Metrics")
#                 st.write(f"MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

#                 st.session_state['model_results']['Transformer'] = {
#                     'forecast': full_forecast,
#                     'metrics': {'mse': mse, 'mae': mae, 'rmse': rmse},
#                     'test_index': btc_data.index[train_size:train_size + len(test_forecast)],
#                     'future_index': pd.date_range(start=btc_data.index[-1], periods=prediction_ahead + 1, freq='D')[1:]
#                 }

#                 last_price = float(btc_data[target_column].iloc[-1])
#                 last_predicted_price = float(forecast[-1])

#                 col1, col2, col3 = st.columns([1, 2, 1])
#                 with col2:
#                     st.markdown(
#                         f"""
#                         <div style="display: flex; justify-content: space-around;">
#                             <div class="stCard greenCard">
#                                 <h3>Latest {target_column}</h3>
#                                 <p style="font-size: 20px;">${last_price:,.2f}</p>
#                             </div>
#                             <div class="stCard greenCard">
#                                 <h3>{target_column} After {prediction_ahead} Days</h3>
#                                 <p style="font-size: 20px;">${last_predicted_price:,.2f}</p>
#                             </div>
#                         </div>
#                         """,
#                         unsafe_allow_html=True,
#                     )

#                 plt.figure(figsize=(14, 5))
#                 plt.plot(btc_data.index, btc_data[target_column], label='Actual', color='blue')
#                 plt.axvline(x=btc_data.index[train_size], color='gray', linestyle='--', label='Train/Test Split')
#                 test_range = btc_data.index[train_size:train_size + len(test_forecast)]
#                 plt.plot(test_range, test_forecast, label='Test Predictions', color='orange')
#                 future_index = pd.date_range(start=btc_data.index[-1], periods=prediction_ahead + 1, freq='D')[1:]
#                 plt.plot(future_index, forecast, label=f'{prediction_ahead}-Day Forecast', color='red')
#                 plt.title(f'{crypto_symbol} Transformer Model Predictions ({target_column})')
#                 plt.xlabel('Date')
#                 plt.ylabel(target_column)
#                 plt.legend()
#                 plt.grid(True)
#                 st.pyplot(plt)




import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_utils import get_yfinance_data, calculate_metrics
from backend.predictor import Predictor

def transformer_forecast_tab(crypto_symbol, period, interval, target_column, prediction_ahead):
    st.header("Transformer Forecasting Model")
    predictor = Predictor()
    try:
        predictor.load_models()
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.warning("Please ensure models are trained by running 'python backend/train_models.py'.")
        return

    if st.button("Predict with Transformers"):
        with st.spinner("Making Transformer predictions..."):
            try:
                btc_data = get_yfinance_data(crypto_symbol, period, interval)
                if btc_data is not None:
                    train_size = int(len(btc_data) * 0.8)
                    train, test = btc_data[:train_size], btc_data[train_size:]

                    forecast = predictor.get_predictions(
                        symbol=crypto_symbol,
                        period=period,
                        interval=interval,
                        target_column=target_column,
                        steps=len(test) + prediction_ahead,
                        model_type="transformer"
                    )

                    test_predictions = forecast[:len(test)]
                    mse, mae, rmse = calculate_metrics(test[target_column], test_predictions)
                    st.subheader("Model Metrics")
                    st.write(f"MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

                    st.session_state['model_results']['Transformer'] = {
                        'forecast': forecast,
                        'metrics': {'mse': mse, 'mae': mae, 'rmse': rmse},
                        'test_index': test.index,
                        'future_index': pd.date_range(start=test.index[-1], periods=prediction_ahead + 1, freq='D')[1:]
                    }

                    latest_price = float(btc_data[target_column].iloc[-1])
                    last_predicted_price = float(forecast[-1])

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
                    plt.plot(train.index, train[target_column], label='Train Data', color='green')
                    plt.plot(test.index, forecast[:len(test)], label='Test Predictions', color='orange')
                    future_index = pd.date_range(start=test.index[-1], periods=prediction_ahead + 1, freq='D')[1:]
                    plt.plot(future_index, forecast[len(test):], label=f'{prediction_ahead}-Day Forecast', color='red')
                    plt.title(f'{crypto_symbol} Transformer Model Predictions ({target_column})')
                    plt.xlabel('Date')
                    plt.ylabel(target_column)
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)

                    forecast_df = pd.DataFrame({
                        'Date': list(test.index) + list(future_index),
                        'Forecast': list(forecast)
                    })
                    csv = forecast_df.to_csv(index=False)
                    st.download_button("Download Forecast", csv, "transformer_forecast.csv", "text/csv")
                else:
                    st.error("Failed to fetch data for prediction.")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")