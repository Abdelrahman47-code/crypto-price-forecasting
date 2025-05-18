import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from itertools import product
import joblib
import os
from datetime import timedelta
from data_utils import get_local_data, ensure_stationarity_multivariate

def train_and_save_models(file_path, interval, target_column="Close", lookback_days=365):
    # Create model directory for the interval
    model_dir = f"backend/models/{interval}"
    os.makedirs(model_dir, exist_ok=True)

    # Fetch data
    try:
        data = get_local_data(file_path, interval)
        if data is None or data.empty:
            raise ValueError(f"No data loaded from {file_path}.")
    except Exception as e:
        print(f"Error loading data for {interval}: {str(e)}")
        return

    # Validate index
    if not isinstance(data.index, pd.DatetimeIndex):
        print(f"Data for {interval} does not have a datetime index.")
        return

    # Filter recent data
    try:
        end_date = data.index.max()
        start_date = end_date - timedelta(days=lookback_days)
        data = data.loc[start_date:end_date]
        if data.empty:
            print(f"No data available for {interval} in the last {lookback_days} days.")
            return
        print(f"Filtered data for {interval}: {len(data)} rows from {data.index.min()} to {data.index.max()}")
    except Exception as e:
        print(f"Error filtering recent data for {interval}: {str(e)}")
        return

    # Define features for multivariate models
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume']

    # Check for missing features
    available_features = [f for f in features if f in data.columns]
    if not available_features:
        print(f"No valid features available for {interval}. Required: {features}")
        return
    if target_column not in data.columns:
        print(f"Target column {target_column} not found in data for {interval}.")
        return

    # Ensure stationarity for VAR and VARMA
    try:
        stationary_data, diff_orders = ensure_stationarity_multivariate(data[available_features], available_features)
        joblib.dump(diff_orders, os.path.join(model_dir, f"diff_orders_{interval}.pkl"))
    except Exception as e:
        print(f"Error ensuring stationarity for {interval}: {str(e)}")
        return

    # Split the data
    train_size = int(len(stationary_data) * 0.9)
    if train_size < 50:  # Ensure enough data for training
        print(f"Insufficient training data for {interval}: {train_size} rows. Need at least 50.")
        return
    train, test = stationary_data[:train_size], stationary_data[train_size:]
    print(f"Training data size: {len(train)}, Test data size: {len(test)}")

    # Train ARIMA
    try:
        p_values = range(0, 5)
        d_values = range(0, 4)
        q_values = range(0, 5)

        def evaluate_arima_model(train, test, arima_order):
            try:
                model = ARIMA(train, order=arima_order)
                model_fit = model.fit()
                predictions = model_fit.forecast(steps=len(test))
                mse = mean_squared_error(test, predictions)
                return mse, model_fit
            except:
                return float('inf'), None

        results = []
        for p, d, q in product(p_values, d_values, q_values):
            arima_order = (p, d, q)
            mse, model_fit = evaluate_arima_model(train[target_column], test[target_column], arima_order)
            if model_fit is not None:
                results.append((arima_order, mse, model_fit))

        if results:
            best_order, best_mse, best_model = min(results, key=lambda x: x[1])
            joblib.dump(best_model, os.path.join(model_dir, f"arima_model_{interval}.pkl"))
            print(f"ARIMA model trained for {interval}. Best order: {best_order}, Test MSE: {best_mse}")
        else:
            print(f"No valid ARIMA models trained for {interval}.")
    except Exception as e:
        print(f"Error training ARIMA model for {interval}: {str(e)}")

    # Train SARIMA
    try:
        seasonal_period = {'15m': 96, '1h': 24, '4h': 6, '1d': 7}[interval]
        results = []
        for p, d, q in product(range(0, 3), range(0, 3), range(0, 3)):
            order = (p, d, q)
            seasonal_order = (p, d, q, seasonal_period)
            try:
                model = SARIMAX(train[target_column], order=order, seasonal_order=seasonal_order)
                model_fit = model.fit(disp=False)
                predictions = model_fit.forecast(steps=len(test))
                mse = mean_squared_error(test[target_column], predictions)
                results.append((order, mse, model_fit))
            except:
                continue

        if results:
            best_order, best_mse, best_model = min(results, key=lambda x: x[1])
            joblib.dump(best_model, os.path.join(model_dir, f"sarima_model_{interval}.pkl"))
            print(f"SARIMA model trained for {interval}. Best order: {best_order}, Test MSE: {best_mse}")
        else:
            print(f"No valid SARIMA models trained for {interval}.")
    except Exception as e:
        print(f"Error training SARIMA model for {interval}: {str(e)}")

    # Train VAR
    try:
        max_lags = min(5, len(train) // 10, len(train) - 1)  # Ensure lags don't exceed data size
        if max_lags < 1:
            print(f"Insufficient data for VAR in {interval}: max_lags={max_lags}")
        else:
            model = VAR(train[available_features])
            results = model.fit(maxlags=max_lags, ic='aic')
            predictions = results.forecast(train[available_features].values[-results.k_ar:], steps=len(test))
            mse = mean_squared_error(test[target_column], predictions[:, available_features.index(target_column)])
            joblib.dump(results, os.path.join(model_dir, f"var_model_{interval}.pkl"))
            print(f"VAR model trained for {interval}. Best lag: {results.k_ar}, Test MSE: {mse}")
    except Exception as e:
        print(f"Error training VAR model for {interval}: {str(e)}")

    # Train VARMA
    try:
        p_values = range(0, 3)
        q_values = range(0, 3)
        results = []
        for p, q in product(p_values, q_values):
            try:
                model = VARMAX(train[available_features], order=(p, q))
                model_fit = model.fit(disp=False)
                predictions = model_fit.forecast(steps=len(test))
                mse = mean_squared_error(test[target_column], predictions[target_column])
                results.append(((p, q), mse, model_fit))
            except:
                continue
        
        if results:
            best_order, best_mse, best_model = min(results, key=lambda x: x[1])
            joblib.dump(best_model, os.path.join(model_dir, f"varma_model_{interval}.pkl"))
            print(f"VARMA model trained for {interval}. Best order: {best_order}, Test MSE: {best_mse}")
        else:
            print(f"No valid VARMA models trained for {interval}.")
    except Exception as e:
        print(f"Error training VARMA model for {interval}: {str(e)}")

    # Train GARCH using arch
    try:
        returns = data[target_column].pct_change().dropna() * 100
        train_returns = returns[:train_size]
        test_returns = returns[train_size:train_size + len(test)]
        model = arch_model(train_returns, vol='Garch', p=1, q=1, mean='AR', lags=1)
        model_fit = model.fit(disp='off')
        forecasts = model_fit.forecast(horizon=len(test))
        predicted_returns = forecasts.mean.values[-1, :]
        mse = mean_squared_error(test_returns[-len(predicted_returns):], predicted_returns)
        joblib.dump(model_fit, os.path.join(model_dir, f"garch_model_{interval}.pkl"))
        print(f"GARCH model trained for {interval}. Test MSE: {mse}")
    except Exception as e:
        print(f"Error training GARCH model for {interval}: {str(e)}")

    # Train LSTM
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[available_features])
        train_data = scaled_data[:train_size]
        time_step = min({'15m': 96, '1h': 24, '4h': 6, '1d': 30}[interval], len(train_data) - 1)
        if time_step < 10:
            print(f"Insufficient data for LSTM in {interval}: time_step={time_step}")
        else:
            X_train, y_train = [], []
            for i in range(len(train_data) - time_step):
                X_train.append(train_data[i:(i + time_step), :])
                y_train.append(train_data[i + time_step, available_features.index(target_column)])
            X_train, y_train = np.array(X_train), np.array(y_train)

            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(time_step, len(available_features))))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)
            joblib.dump(scaler, os.path.join(model_dir, f"lstm_scaler_{interval}.pkl"))
            model.save(os.path.join(model_dir, f"lstm_model_{interval}.h5"))
            print(f"LSTM model trained for {interval}. Training samples: {X_train.shape[0]}")
    except Exception as e:
        print(f"Error training LSTM model for {interval}: {str(e)}")

if __name__ == "__main__":
    csv_files = {
        # "15m": "data/btc_15m_data_2018_to_2025.csv",
        # "1h": "data/btc_1h_data_2018_to_2025.csv",
        # "4h": "data/btc_4h_data_2018_to_2025.csv",
        "1d": "data/btc_1d_data_2018_to_2025.csv"
    }
    lookback_days = 1000
    for interval, file_path in csv_files.items():
        print(f"\nTraining models for interval: {interval}")
        train_and_save_models(file_path, interval, lookback_days=lookback_days)