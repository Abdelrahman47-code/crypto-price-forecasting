import streamlit as st
import pandas as pd
import plotly.express as px
from backend.data_utils import get_local_data, calculate_metrics
from backend.predictor import Predictor
from datetime import timedelta

def model_comparison_tab(file_path, interval, target_column, prediction_ahead, lookback_days):
    st.header(f"Model Comparison ({interval})")

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

        # Initialize predictor
        predictor = Predictor(interval=interval)
        predictor.load_models()

        # Define models
        models = ["ARIMA", "SARIMA", "VAR", "VARMA", "GARCH", "LSTM"]

        # Generate forecasts and metrics
        results = []
        combined_df = data[[target_column]].reset_index().rename(
            columns={'Open time': 'Date', target_column: 'Price'}
        ).assign(Model='Historical')

        # Create future index for forecasts
        last_date = data.index[-1]
        freq = {'15m': '15min', '1h': 'H', '4h': '4H', '1d': 'D'}[interval]
        future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_ahead, freq=freq)

        # Test data for metrics
        test_size = int(len(data) * 0.1)
        test_data = data[-test_size:][target_column] if test_size > 0 else None

        for model in models:
            try:
                with st.spinner(f"Generating {model} forecast..."):
                    # Get forecast
                    time_step = {'15m': 96, '1h': 24, '4h': 6, '1d': 30}[interval]
                    forecast = predictor.get_predictions(
                        file_path=file_path,
                        interval=interval,
                        target_column=target_column,
                        steps=prediction_ahead,
                        model_type=model.lower(),
                        time_step=time_step
                    )

                    # Store forecast in session state
                    st.session_state['model_results'][model]['forecast'] = forecast
                    st.session_state['model_results'][model]['future_index'] = future_index

                    # Create forecast DataFrame for plotting
                    forecast_df = pd.DataFrame({
                        'Date': future_index,
                        'Price': forecast,
                        'Model': model
                    })
                    combined_df = pd.concat([combined_df, forecast_df], ignore_index=True)

                    # Calculate metrics
                    metrics = None
                    if test_size > 0:
                        test_forecast = predictor.get_predictions(
                            file_path=file_path,
                            interval=interval,
                            target_column=target_column,
                            steps=test_size,
                            model_type=model.lower(),
                            time_step=time_step
                        )
                        mse, mae, rmse = calculate_metrics(test_data, test_forecast[:len(test_data)])
                        metrics = {'MSE': mse, 'MAE': mae, 'RMSE': rmse}
                        st.session_state['model_results'][model]['metrics'] = metrics
                    else:
                        metrics = {'MSE': None, 'MAE': None, 'RMSE': None}

                    results.append({
                        'Model': model,
                        'MSE': metrics['MSE'],
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE']
                    })

            except Exception as e:
                st.warning(f"Error generating {model} forecast: {str(e)}")
                results.append({
                    'Model': model,
                    'MSE': None,
                    'MAE': None,
                    'RMSE': None
                })

        # Display metrics table
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame(results)
        st.dataframe(
            metrics_df.style.format(
                {
                    'MSE': '{:.2f}' if metrics_df['MSE'].notnull().any() else '',
                    'MAE': '{:.2f}' if metrics_df['MAE'].notnull().any() else '',
                    'RMSE': '{:.2f}' if metrics_df['RMSE'].notnull().any() else ''
                },
                na_rep='N/A'
            ),
            use_container_width=True
        )
        st.write("Note: Lower MSE, MAE, and RMSE indicate better model performance. Metrics are calculated on the last 10% of historical data.")

        # Plot all forecasts
        st.subheader("Forecast Comparison")
        fig = px.line(
            combined_df,
            x='Date',
            y='Price',
            color='Model',
            title=f'Historical and Forecasted {target_column} ({interval})',
            labels={'Date': 'Date', 'Price': target_column}
        )
        fig.update_traces(
            selector=dict(name='Historical'),
            line=dict(width=3)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download combined forecast
        forecast_df = pd.DataFrame({'Date': future_index})
        for model in models:
            forecast = st.session_state['model_results'][model]['forecast']
            if forecast is not None:
                forecast_df[model] = forecast
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            "Download Combined Forecast",
            csv,
            f"combined_forecast_{interval}.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error in model comparison for {interval}: {str(e)}")