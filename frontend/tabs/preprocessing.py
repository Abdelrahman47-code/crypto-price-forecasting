import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from backend.data_utils import get_local_data, check_stationarity, plot_acf_pacf
from datetime import timedelta

def preprocessing_tab(file_path, interval, target_column, lookback_days):
    st.header(f"Data Preprocessing ({interval})")

    try:
        # Load data
        data = get_local_data(file_path, interval)
        if data is None or data.empty:
            st.error("Failed to load data from CSV.")
            return

        # Filter by lookback period
        end_date = data.index.max()
        start_date = end_date - timedelta(days=lookback_days)
        data = data.loc[start_date:end_date]

        # Store filtered data in session state
        st.session_state['btc_data'] = data

        # Display raw data
        st.subheader("Raw Data")
        st.write(data)

        # Plot target column
        st.subheader(f"{target_column} Price Over Time")
        fig = px.line(
            data,
            x=data.index,
            y=target_column,
            title=f"{target_column} Price Over Time ({interval})",
            labels={'x': 'Date', 'y': target_column}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature correlation heatmap
        st.subheader("Feature Correlations")
        corr_matrix = data.corr()
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            annotation_text=corr_matrix.round(2).values,
            colorscale='Viridis',
            showscale=True
        )
        fig.update_layout(
            title=f"Correlation Matrix of Features ({interval})",
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,
            height=800
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("Note: High correlations (e.g., >0.9) between features like 'Volume' and 'Quote asset volume' may indicate redundancy.")

        # Stationarity check for target column
        st.subheader(f"Stationarity Check for {target_column}")
        is_stationary, p_value, _ = check_stationarity(data, target_column)
        st.write(f"Augmented Dickey-Fuller Test p-value: {p_value:.4f}")
        st.write(f"Is {target_column} stationary? {'Yes' if is_stationary else 'No'}")

        # ACF and PACF plots
        st.subheader("ACF and PACF Plots")
        fig = plot_acf_pacf(data[target_column], lags=40)
        st.pyplot(fig)
        st.write("ACF/PACF plots help identify ARIMA/SARIMA orders. Significant lags in ACF suggest MA terms, while PACF suggests AR terms.")

    except Exception as e:
        st.error(f"Error in preprocessing for {interval}: {str(e)}")