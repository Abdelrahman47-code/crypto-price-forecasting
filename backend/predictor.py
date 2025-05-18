import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from backend.data_utils import get_local_data, ensure_stationarity_multivariate

class Predictor:
    def __init__(self, interval, model_dir="backend/models"):
        self.interval = interval
        self.model_dir = os.path.join(model_dir, interval)
        self.arima_model = None
        self.lstm_model = None
        self.sarima_model = None
        self.garch_model = None
        self.var_model = None
        self.varma_model = None
        self.lstm_scaler = None
        self.diff_orders = None
        self.loaded_models = set()

    def load_models(self):
        errors = []

        # Load ARIMA model
        arima_path = os.path.join(self.model_dir, f"arima_model_{self.interval}.pkl")
        if os.path.exists(arima_path):
            try:
                self.arima_model = joblib.load(arima_path)
                self.loaded_models.add("arima")
            except Exception as e:
                errors.append(f"Failed to load ARIMA model for {self.interval}: {str(e)}")

        # Load SARIMA model
        sarima_path = os.path.join(self.model_dir, f"sarima_model_{self.interval}.pkl")
        if os.path.exists(sarima_path):
            try:
                self.sarima_model = joblib.load(sarima_path)
                self.loaded_models.add("sarima")
            except Exception as e:
                errors.append(f"Failed to load SARIMA model for {self.interval}: {str(e)}")

        # Load VAR model
        var_path = os.path.join(self.model_dir, f"var_model_{self.interval}.pkl")
        if os.path.exists(var_path):
            try:
                self.var_model = joblib.load(var_path)
                self.loaded_models.add("var")
            except Exception as e:
                errors.append(f"Failed to load VAR model for {self.interval}: {str(e)}")

        # Load VARMA model
        varma_path = os.path.join(self.model_dir, f"varma_model_{self.interval}.pkl")
        if os.path.exists(varma_path):
            try:
                self.varma_model = joblib.load(varma_path)
                self.loaded_models.add("varma")
            except Exception as e:
                errors.append(f"Failed to load VARMA model for {self.interval}: {str(e)}")

        # Load differencing orders
        diff_orders_path = os.path.join(self.model_dir, f"diff_orders_{self.interval}.pkl")
        if os.path.exists(diff_orders_path):
            try:
                self.diff_orders = joblib.load(diff_orders_path)
            except Exception as e:
                errors.append(f"Failed to load differencing orders for {self.interval}: {str(e)}")

        # Load GARCH model
        garch_path = os.path.join(self.model_dir, f"garch_model_{self.interval}.pkl")
        if os.path.exists(garch_path):
            try:
                self.garch_model = joblib.load(garch_path)
                self.loaded_models.add("garch")
            except Exception as e:
                errors.append(f"Failed to load GARCH model for {self.interval}: {str(e)}")

        # Load LSTM model and scaler
        lstm_path = os.path.join(self.model_dir, f"lstm_model_{self.interval}.h5")
        lstm_scaler_path = os.path.join(self.model_dir, f"lstm_scaler_{self.interval}.pkl")
        if os.path.exists(lstm_path) and os.path.exists(lstm_scaler_path):
            try:
                self.lstm_model = load_model(lstm_path)
                self.lstm_scaler = joblib.load(lstm_scaler_path)
                self.loaded_models.add("lstm")
            except Exception as e:
                errors.append(f"Failed to load LSTM model or scaler for {self.interval}: {str(e)}")

        if errors:
            raise Exception("\n".join(errors))
        if not self.loaded_models:
            raise Exception(f"No models were loaded successfully for {self.interval}. Please ensure models are trained for this interval.")

    def fetch_data(self, file_path, interval):
        try:
            data = get_local_data(file_path, interval)
            if data is None or data.empty:
                raise ValueError("No data loaded from CSV.")
            return data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def predict_arima(self, data, target_column, steps):
        if self.arima_model is None:
            raise ValueError("ARIMA model not loaded.")
        history = data[target_column].values
        forecast = self.arima_model.forecast(steps=steps)
        return forecast

    def predict_sarima(self, data, target_column, steps):
        if self.sarima_model is None:
            raise ValueError("SARIMA model not loaded.")
        history = data[target_column].values
        forecast = self.sarima_model.forecast(steps=steps)
        return forecast

    def predict_var(self, data, target_column, steps):
        if self.var_model is None or self.diff_orders is None:
            raise ValueError("VAR model or differencing orders not loaded.")
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Quote asset volume', 'Number of trades',
                    'Taker buy base asset volume', 'Taker buy quote asset volume']
        available_features = [f for f in features if f in data.columns]
        if target_column not in available_features:
            raise ValueError(f"Target column {target_column} not in available features: {available_features}")
        
        # Ensure stationarity
        stationary_data, _ = ensure_stationarity_multivariate(data[available_features], available_features)
        if len(stationary_data) < self.var_model.k_ar:
            raise ValueError(f"Insufficient data for VAR forecasting. Need at least {self.var_model.k_ar} rows, got {len(stationary_data)}.")
        
        # Generate forecast
        forecast = self.var_model.forecast(stationary_data[available_features].values[-self.var_model.k_ar:], steps=steps)
        target_idx = available_features.index(target_column)
        forecast = forecast[:, target_idx]
        
        # Reverse differencing if needed
        if self.diff_orders[target_column] > 0:
            last_value = data[target_column].iloc[-1]
            diff_order = self.diff_orders[target_column]
            # Integrate the differenced forecast
            forecast = np.cumsum(forecast) + last_value
            # Trim to steps if necessary
            forecast = forecast[:steps]
        
        print(f"VAR forecast for {self.interval}: {len(forecast)} steps requested, {len(forecast)} returned")
        return forecast

    def predict_varma(self, data, target_column, steps):
        if self.varma_model is None or self.diff_orders is None:
            raise ValueError("VARMA model or differencing orders not loaded.")
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Quote asset volume', 'Number of trades',
                    'Taker buy base asset volume', 'Taker buy quote asset volume']
        available_features = [f for f in features if f in data.columns]
        if target_column not in available_features:
            raise ValueError(f"Target column {target_column} not in available features: {available_features}")
        
        # Ensure stationarity
        stationary_data, _ = ensure_stationarity_multivariate(data[available_features], available_features)
        if len(stationary_data) < 10:
            raise ValueError(f"Insufficient data for VARMA forecasting. Need at least 10 rows, got {len(stationary_data)}.")
        
        # Generate forecast
        forecast_df = self.varma_model.forecast(steps=steps)
        if target_column not in forecast_df.columns:
            raise ValueError(f"Target column {target_column} not in VARMA forecast columns: {forecast_df.columns}")
        forecast = forecast_df[target_column].values
        
        # Reverse differencing if needed
        if self.diff_orders[target_column] > 0:
            last_value = data[target_column].iloc[-1]
            diff_order = self.diff_orders[target_column]
            # Integrate the differenced forecast
            forecast = np.cumsum(forecast) + last_value
            # Trim to steps if necessary
            forecast = forecast[:steps]
        
        print(f"VARMA forecast for {self.interval}: {len(forecast)} steps requested, {len(forecast)} returned")
        return forecast

    def predict_garch(self, data, target_column, steps):
        if self.garch_model is None:
            raise ValueError("GARCH model not loaded.")
        returns = data[target_column].pct_change().dropna() * 100
        forecasts = self.garch_model.forecast(horizon=steps)
        predicted_returns = forecasts.mean.values[-1, :]
        last_price = data[target_column].iloc[-1]
        forecast_prices = last_price * (1 + np.cumsum(predicted_returns / 100))
        return forecast_prices

    def predict_lstm(self, data, target_column, steps, time_step=60):
        if self.lstm_model is None or self.lstm_scaler is None:
            raise ValueError("LSTM model or scaler not loaded.")
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Quote asset volume', 'Number of trades',
                    'Taker buy base asset volume', 'Taker buy quote asset volume']
        scaled_data = self.lstm_scaler.transform(data[features])
        last_data = scaled_data[-time_step:]
        current_batch = last_data.reshape((1, time_step, len(features)))
        future_forecast = []
        for _ in range(steps):
            pred = self.lstm_model.predict(current_batch, verbose=0)
            future_forecast.append(pred[0, 0])
            new_row = np.zeros((1, len(features)))
            new_row[0, features.index(target_column)] = pred[0, 0]
            current_batch = np.append(current_batch[:, 1:, :], [new_row], axis=1)
        future_forecast = np.array(future_forecast).reshape(-1, 1)
        dummy = np.zeros((len(future_forecast), len(features)))
        dummy[:, features.index(target_column)] = future_forecast.flatten()
        future_forecast = self.lstm_scaler.inverse_transform(dummy)[:, features.index(target_column)]
        return future_forecast
    
    def get_predictions(self, file_path, interval, target_column, steps, model_type="arima", time_step=60):
        if model_type.lower() not in self.loaded_models:
            raise ValueError(f"Model type '{model_type}' not loaded for {interval}. Available models: {self.loaded_models}")
        data = self.fetch_data(file_path, interval)
        if data is None or data.empty:
            raise ValueError("No data available for prediction.")
        if model_type.lower() == "arima":
            return self.predict_arima(data, target_column, steps)
        elif model_type.lower() == "sarima":
            return self.predict_sarima(data, target_column, steps)
        elif model_type.lower() == "garch":
            return self.predict_garch(data, target_column, steps)
        elif model_type.lower() == "lstm":
            return self.predict_lstm(data, target_column, steps, time_step)
        elif model_type.lower() == "var":
            return self.predict_var(data, target_column, steps)
        elif model_type.lower() == "varma":
            return self.predict_varma(data, target_column, steps)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")