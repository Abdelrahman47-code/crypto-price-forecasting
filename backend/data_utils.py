import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_local_data(file_path, interval):
    try:
        # Load data from local CSV file
        data = pd.read_csv(file_path)
        print(f"Processing local file: {file_path}")

        # Strip whitespace from column names
        data.columns = [col.strip() for col in data.columns]
        print(f"Columns after stripping whitespace: {data.columns.tolist()}")

        # Validate that there are enough columns
        if len(data.columns) < 5:  # Need at least timestamp, Open, High, Low, Close
            raise ValueError("CSV must have at least 5 columns: timestamp, Open, High, Low, Close")

        # Use first column as the timestamp
        timestamp_col = data.columns[0]
        print(f"Using first column as timestamp: {timestamp_col}")

        # Validate required columns (excluding timestamp)
        required_columns = ['Open', 'High', 'Low', 'Close']
        optional_columns = ['Volume', 'Quote asset volume', 'Number of trades',
                           'Taker buy base asset volume', 'Taker buy quote asset volume', 'Close time']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {missing_columns}")

        # Convert timestamp (first column) to datetime
        if pd.api.types.is_numeric_dtype(data[timestamp_col]):
            try:
                data[timestamp_col] = pd.to_datetime(data[timestamp_col], unit='ms', errors='raise')
            except Exception as e:
                raise ValueError(f"Invalid timestamp (column {timestamp_col}) numeric format: {str(e)}. Expected milliseconds.")
        else:
            try:
                data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors='raise')
            except Exception as e:
                raise ValueError(f"Invalid timestamp (column {timestamp_col}) date format: {str(e)}. Expected YYYY-MM-DD (for 1d) or YYYY-MM-DD HH:MM:SS (for 15m, 1h, 4h).")

        # Set index to the timestamp column
        data.set_index(timestamp_col, inplace=True)

        # Drop Close time if present
        if 'Close time' in data.columns:
            data = data.drop(columns=['Close time'])

        # Validate interval consistency
        expected_freq = {'15m': '15min', '1h': 'H', '4h': '4H', '1d': 'D'}[interval]
        inferred_freq = pd.infer_freq(data.index[:10])
        if inferred_freq and inferred_freq.upper() != expected_freq.upper():
            raise ValueError(f"Data interval mismatch. Expected {expected_freq}, got {inferred_freq or 'unknown'}.")

        # Ensure numeric columns
        for col in required_columns + optional_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Drop rows with missing values
        data.dropna(subset=required_columns, inplace=True)

        return data

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def check_stationarity(data, target_column):
    try:
        series = data[target_column].dropna()
        result = adfuller(series)
        p_value = result[1]
        is_stationary = p_value < 0.05
        test_statistic = result[0]
        return is_stationary, p_value, test_statistic
    except Exception as e:
        print(f"Error in stationarity check: {str(e)}")
        return False, None, None

def plot_acf_pacf(series, lags=40):
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(series.dropna(), lags=lags, ax=ax1)
        plot_pacf(series.dropna(), lags=lags, ax=ax2)
        ax1.set_title('Autocorrelation Function (ACF)')
        ax2.set_title('Partial Autocorrelation Function (PACF)')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error plotting ACF/PACF: {str(e)}")
        plt.close()
        return None

def calculate_metrics(actual, predicted):
    try:
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)
        return mse, mae, rmse
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None, None, None

def ensure_stationarity_multivariate(data, columns):
    try:
        stationary_data = data[columns].copy()
        diff_orders = {col: 0 for col in columns}
        for col in columns:
            series = data[col].dropna()
            is_stationary, _, _ = check_stationarity(data, col)
            diff_count = 0
            while not is_stationary and diff_count < 2:
                series = series.diff().dropna()
                diff_count += 1
                is_stationary, _, _ = check_stationarity(series, col)
            diff_orders[col] = diff_count
            if diff_count > 0:
                stationary_data[col] = data[col].diff(periods=diff_count).dropna()
        stationary_data.dropna(inplace=True)
        return stationary_data, diff_orders
    except Exception as e:
        print(f"Error ensuring stationarity: {str(e)}")
        return data, {col: 0 for col in columns}