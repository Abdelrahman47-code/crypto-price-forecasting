import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_yfinance_data(symbol, period, interval):
    try:
        data = yf.download(symbol, period=period, interval=interval)
        return data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    except Exception as e:
        return None

def check_stationarity(data, column):
    result = adfuller(data[column])
    p_value = result[1]
    return p_value < 0.05, p_value, result

def make_stationary(data, column):
    data_diff = data[column].diff().dropna()
    return data_diff

def plot_acf_pacf(data, lags=40):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(data, lags=lags, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)')
    plot_pacf(data, lags=lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    return fig

def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    return mse, mae, rmse