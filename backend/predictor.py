import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
import joblib
import os

class Predictor:
    def __init__(self, model_dir="backend/models"):
        self.model_dir = model_dir
        self.arima_model = None
        self.lstm_model = None
        self.transformer_model = None
        self.sarima_model = None
        self.garch_mean_model = None
        self.loaded_models = set()

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Define the transformer encoder layer for custom object loading."""
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x + inputs)
        x_ff = Dense(ff_dim, activation="relu")(x)
        x_ff = Dense(inputs.shape[-1])(x_ff)
        x_ff = Dropout(dropout)(x_ff)
        x = LayerNormalization(epsilon=1e-6)(x + x_ff)
        return x

    def load_models(self):
        """Load pre-trained models from the models directory."""
        errors = []

        # Load ARIMA model
        arima_path = os.path.join(self.model_dir, "arima_model.pkl")
        if os.path.exists(arima_path):
            try:
                self.arima_model = joblib.load(arima_path)
                self.loaded_models.add("arima")
            except Exception as e:
                errors.append(f"Failed to load ARIMA model: {str(e)}")
        else:
            errors.append(f"ARIMA model file not found at {arima_path}")

        # Load SARIMA model
        sarima_path = os.path.join(self.model_dir, "sarima_model.pkl")
        if os.path.exists(sarima_path):
            try:
                self.sarima_model = joblib.load(sarima_path)
                self.loaded_models.add("sarima")
            except Exception as e:
                errors.append(f"Failed to load SARIMA model: {str(e)}")
        else:
            errors.append(f"SARIMA model file not found at {sarima_path}")

        # Load GARCH mean model
        garch_mean_path = os.path.join(self.model_dir, "garch_mean_model.pkl")
        if os.path.exists(garch_mean_path):
            try:
                self.garch_mean_model = joblib.load(garch_mean_path)
                self.loaded_models.add("garch")
            except Exception as e:
                errors.append(f"Failed to load GARCH mean model: {str(e)}")
        else:
            errors.append(f"GARCH mean model file not found at {garch_mean_path}")

        # Load LSTM model
        lstm_path = os.path.join(self.model_dir, "lstm_model.h5")
        if os.path.exists(lstm_path):
            try:
                self.lstm_model = load_model(lstm_path)
                self.loaded_models.add("lstm")
            except Exception as e:
                errors.append(f"Failed to load LSTM model: {str(e)}")
        else:
            errors.append(f"LSTM model file not found at {lstm_path}")

        # # Load Transformer model with custom objects
        # transformer_path = os.path.join(self.model_dir, "transformer_model.h5")
        # if os.path.exists(transformer_path):
        #     try:
        #         # Define custom objects for the Transformer model
        #         custom_objects = {
        #             'MultiHeadAttention': MultiHeadAttention,
        #             'LayerNormalization': LayerNormalization
        #         }
        #         self.transformer_model = load_model(transformer_path, custom_objects=custom_objects)
        #         self.loaded_models.add("transformer")
        #     except Exception as e:
        #         errors.append(f"Failed to load Transformer model: {str(e)}")
        # else:
        #     errors.append(f"Transformer model file not found at {transformer_path}")

        if errors:
            raise Exception("\n".join(errors))
        if not self.loaded_models:
            raise Exception("No models were loaded successfully.")

    def fetch_data(self, symbol, period, interval):
        """Fetch data using yfinance."""
        try:
            data = yf.download(symbol, period=period, interval=interval)
            return data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")

    def predict_arima(self, data, target_column, steps):
        """Make predictions using the ARIMA model."""
        if self.arima_model is None:
            raise ValueError("ARIMA model not loaded.")
        history = data[target_column].values
        forecast = self.arima_model.forecast(steps=steps)
        return forecast

    def predict_sarima(self, data, target_column, steps):
        """Make predictions using the SARIMA model."""
        if self.sarima_model is None:
            raise ValueError("SARIMA model not loaded.")
        history = data[target_column].values
        forecast = self.sarima_model.forecast(steps=steps)
        return forecast

    def predict_garch(self, data, target_column, steps):
        """Make predictions using the GARCH model (mean model for price)."""
        if self.garch_mean_model is None:
            raise ValueError("GARCH mean model not loaded.")
        returns = data[target_column].pct_change().dropna() * 100
        forecast = self.garch_mean_model.forecast(steps=steps)
        return forecast

    def predict_lstm(self, data, target_column, steps, time_step=60):
        """Make predictions using the LSTM model."""
        if self.lstm_model is None:
            raise ValueError("LSTM model not loaded.")
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[[target_column]])

        last_data = scaled_data[-time_step:]
        current_batch = last_data.reshape((1, time_step, 1))

        future_forecast = []
        for _ in range(steps):
            pred = self.lstm_model.predict(current_batch, verbose=0)
            future_forecast.append(pred[0, 0])
            current_batch = np.append(current_batch[:, 1:, :], [[[pred[0, 0]]]], axis=1)

        future_forecast = np.array(future_forecast).reshape(-1, 1)
        future_forecast = scaler.inverse_transform(future_forecast)
        return future_forecast.flatten()

    def predict_transformer(self, data, target_column, steps, time_step=60):
        """Make predictions using the Transformer model."""
        if self.transformer_model is None:
            raise ValueError("Transformer model not loaded.")
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[[target_column]])

        last_data = scaled_data[-time_step:]
        current_batch = last_data.reshape((1, time_step, 1))

        future_forecast = []
        for _ in range(steps):
            pred = self.transformer_model.predict(current_batch, verbose=0)
            future_forecast.append(pred[0, 0])
            current_batch = np.append(current_batch[:, 1:, :], [[[pred[0, 0]]]], axis=1)

        future_forecast = np.array(future_forecast).reshape(-1, 1)
        future_forecast = scaler.inverse_transform(future_forecast)
        return future_forecast.flatten()

    def get_predictions(self, symbol, period, interval, target_column, steps, model_type="arima", time_step=60):
        """Fetch data and make predictions using the specified model."""
        if model_type.lower() not in self.loaded_models:
            raise ValueError(f"Model type '{model_type}' not loaded. Available models: {self.loaded_models}")
        data = self.fetch_data(symbol, period, interval)
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
        # elif model_type.lower() == "transformer":
        #     return self.predict_transformer(data, target_column, steps, time_step)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")