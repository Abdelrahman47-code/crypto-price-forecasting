import streamlit as st
import matplotlib.pyplot as plt
from utils.data_utils import get_yfinance_data, check_stationarity, make_stationary, plot_acf_pacf

def preprocessing_tab(crypto_symbol, period, interval, target_column):
    st.header("Data Preprocessing and Analysis")
    
    if st.button("Analyze Data"):
        with st.spinner("Fetching data..."):
            btc_data = get_yfinance_data(crypto_symbol, period, interval)
            if btc_data is not None:
                st.session_state['btc_data'] = btc_data
                st.subheader("Raw Data")
                st.write(btc_data)

    if st.session_state['btc_data'] is not None:
        btc_data = st.session_state['btc_data']
        
        st.subheader("Explore Time Series Data")
        column_to_plot = st.selectbox("Select Column to Plot", btc_data.columns)
        if column_to_plot:
            plt.figure(figsize=(14, 5))
            plt.plot(btc_data.index, btc_data[column_to_plot], label=column_to_plot, color='blue')
            plt.title(f'{crypto_symbol} {column_to_plot} Over Time')
            plt.xlabel('Date')
            plt.ylabel(column_to_plot)
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

        if st.button("Run Stationarity Analysis"):
            with st.spinner("Running stationarity analysis..."):
                is_stationary, p_value, adf_result = check_stationarity(btc_data, column=target_column)
                st.subheader("Stationarity Test (Augmented Dickey-Fuller)")
                st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                st.write(f"p-value: {p_value:.4f}")
                st.write("Stationary" if is_stationary else "Non-Stationary")
                
                if not is_stationary:
                    st.subheader("Making Data Stationary")
                    btc_stationary = make_stationary(btc_data, column=target_column)
                    st.write(f"{target_column} has been differenced to achieve stationarity.")
                else:
                    btc_stationary = btc_data[target_column]
                    st.write(f"{target_column} is already stationary.")

                st.subheader("ACF and PACF Plots")
                fig = plot_acf_pacf(btc_stationary)
                st.pyplot(fig)

                st.subheader("Original vs. Stationary Data")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
                ax1.plot(btc_data.index, btc_data[target_column], label='Original Data', color='blue')
                ax1.set_title(f'{crypto_symbol} Original {target_column}')
                ax1.set_xlabel('Date')
                ax1.set_ylabel(target_column)
                ax1.legend()
                ax1.grid(True)
                
                ax2.plot(btc_stationary.index, btc_stationary, label='Stationary Data (Differenced)', color='green')
                ax2.set_title(f'Stationary {target_column}')
                ax2.set_xlabel('Date')
                ax2.set_ylabel(f'Differenced {target_column}')
                ax2.legend()
                ax2.grid(True)
                plt.tight_layout()
                st.pyplot(fig)

