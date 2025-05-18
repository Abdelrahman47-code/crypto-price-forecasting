import streamlit as st
import pandas as pd
import plotly.express as px
from backend.data_utils import get_local_data, calculate_metrics
from backend.predictor import Predictor
from datetime import timedelta

def lstm_forecast_tab(file_path, interval, target_column, prediction_ahead, lookback_days):
    st.header(f"LSTM Forecast ({interval})")
    
    try:
        # Load data
        data = get_local_data(file_path, interval)
        if data is None or data.empty:
            st.error("Failed to load data.")
            return

        # Filter by lookback period
        end_date = data.index.max()
        start_date = end_date - timedelta(days=lookback_days)
        data = data.loc[start_date:end_date]

        # Initialize predictor with interval
        predictor = Predictor(interval=interval)
        predictor.load_models()

        if st.button("Predict with LSTM"):
            with st.spinner("Making LSTM predictions..."):
                # Get predictions
                time_step = {'15m': 96, '1h': 24, '4h': 6, '1d': 30}[interval]
                forecast = predictor.get_predictions(
                    file_path=file_path,
                    interval=interval,
                    target_column=target_column,
                    steps=prediction_ahead,
                    model_type="lstm",
                    time_step=time_step
                )

                # Create future index
                last_date = data.index[-1]
                freq = {'15m': '15min', '1h': 'H', '4h': '4H', '1d': 'D'}[interval]
                future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_ahead, freq=freq)

                # Store results in session state
                st.session_state['model_results']['LSTM']['forecast'] = forecast
                st.session_state['model_results']['LSTM']['future_index'] = future_index

                # Display latest and predicted prices
                latest_price = float(data[target_column].iloc[-1])
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

                # Plot historical data and forecast
                historical_df = data[[target_column]].reset_index().rename(columns={'Open time': 'Date', target_column: 'Price'})
                forecast_df = pd.DataFrame({
                    'Date': future_index,
                    'Price': forecast
                })
                combined_df = pd.concat([
                    historical_df.assign(Type='Historical'),
                    forecast_df.assign(Type='Forecast')
                ])

                fig = px.line(combined_df, x='Date', y='Price', color='Type', title=f'LSTM Forecast for {target_column} ({interval})')
                st.plotly_chart(fig, use_container_width=True)

                # Display metrics
                test_size = int(len(data) * 0.1)
                if test_size > 0:
                    test_data = data[-test_size:][target_column]
                    test_forecast = predictor.get_predictions(
                        file_path=file_path,
                        interval=interval,
                        target_column=target_column,
                        steps=test_size,
                        model_type="lstm",
                        time_step=time_step
                    )
                    mse, mae, rmse = calculate_metrics(test_data, test_forecast[:len(test_data)])
                    st.subheader("Model Performance Metrics")
                    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

                # Download forecast
                forecast_df = pd.DataFrame({
                    'Date': future_index,
                    'Forecast': forecast
                })
                csv = forecast_df.to_csv(index=False)
                st.download_button("Download Forecast", csv, f"lstm_forecast_{interval}.csv", "text/csv")

    except Exception as e:
        st.error(f"Error generating LSTM forecast for {interval}: {str(e)}")