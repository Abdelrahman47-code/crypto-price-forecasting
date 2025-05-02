import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, LSTM
from sklearn.metrics import mean_squared_error
from itertools import product
import joblib
import os

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    x = LayerNormalization(epsilon=1e-6)(x + x_ff)
    return x

def build_transformer_model(time_step, features, head_size=64, num_heads=4, ff_dim=64, num_transformer_blocks=2, dropout=0.1):
    inputs = Input(shape=(time_step, features))
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

def train_and_save_models(symbol="BTC-USD", period="2y", interval="1d", target_column="Close"):
    # Create models directory if it doesn't exist
    model_dir = "backend/models"
    os.makedirs(model_dir, exist_ok=True)

    # Fetch data
    try:
        data = yf.download(symbol, period=period, interval=interval)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        if data.empty:
            raise ValueError("No data fetched from yfinance.")
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return

    # Adjust train-test split for larger dataset (90/10 split to maximize training data)
    train_size = int(len(data) * 0.9)  # ~453 days for training, ~51 for testing with period="2y"
    train, test = data[:train_size], data[train_size:]
    print(f"Training data size: {len(train)}, Test data size: {len(test)}")

    # Train ARIMA
    try:
        # Optimize grid search for larger dataset
        p_values = range(0, 3)  # Reduced range to speed up training
        d_values = range(0, 2)
        q_values = range(0, 3)

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
            results.append((arima_order, mse, model_fit))

        best_order, best_mse, best_model = min(results, key=lambda x: x[1])
        joblib.dump(best_model, os.path.join(model_dir, "arima_model.pkl"))
        print(f"\nARIMA model trained and saved successfully. Best order: {best_order}, Test MSE: {best_mse}\n")
    except Exception as e:
        print(f"Error training ARIMA model: {str(e)}")

    # Train SARIMA
    try:
        seasonal_period = 7  # Weekly seasonality
        results = []
        for p, d, q in product(range(0, 2), range(0, 2), range(0, 2)):  # Reduced range for efficiency
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

        best_order, best_mse, best_model = min(results, key=lambda x: x[1])
        joblib.dump(best_model, os.path.join(model_dir, "sarima_model.pkl"))
        print(f"\nSARIMA model trained and saved successfully. Best order: {best_order}, Test MSE: {best_mse}\n")
    except Exception as e:
        print(f"Error training SARIMA model: {str(e)}")

    # Train GARCH (mean model for price prediction)
    try:
        returns = train[target_column].pct_change().dropna() * 100
        mean_model = ARIMA(returns, order=(1, 0, 0)).fit()
        joblib.dump(mean_model, os.path.join(model_dir, "garch_mean_model.pkl"))
        print("\nGARCH mean model trained and saved successfully.\n")
    except Exception as e:
        print(f"Error training GARCH mean model: {str(e)}")

    # Train LSTM
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[[target_column]])
        train_data = scaled_data[:train_size]
        time_step = 30  # Reduced from 60 to create more training samples
        X_train, y_train = [], []
        for i in range(len(train_data) - time_step):
            X_train.append(train_data[i:(i + time_step), 0])
            y_train.append(train_data[i + time_step, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)  # Increased batch_size and epochs
        model.save(os.path.join(model_dir, "lstm_model.h5"))
        print(f"\nLSTM model trained and saved successfully. Training samples: {X_train.shape[0]}\n")
    except Exception as e:
        print(f"Error training LSTM model: {str(e)}")

    # Transformer model (commented out as per user preference)
    # try:
    #     X_train, y_train = [], []
    #     for i in range(len(train_data) - time_step):
    #         X_train.append(train_data[i:(i + time_step), 0])
    #         y_train.append(train_data[i + time_step, 0])
    #     X_train, y_train = np.array(X_train), np.array(y_train)
    #     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    #     model = build_transformer_model(time_step=time_step, features=1)
    #     model.compile(optimizer='adam', loss='mean_squared_error')
    #     model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)
    #     model.save(os.path.join(model_dir, "transformer_model.h5"))
    #     print("\nTransformer model trained and saved successfully.\n")
    # except Exception as e:
    #     print(f"Error training Transformer model: {str(e)}")

if __name__ == "__main__":
    train_and_save_models()