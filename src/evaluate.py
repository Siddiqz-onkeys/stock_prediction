import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from data_loader import fetch_stock_data, preprocess_data
from predict import predict_future_prices
from sklearn.model_selection import train_test_split


def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    return mean_absolute_percentage_error(y_true, y_pred) * 100

if __name__ == "__main__":
    # Load model and scaler
    model_path = "models/lstm_stock_model.h5"
    ticker = "AMZN"
    stock_data = fetch_stock_data(ticker)
    scaled_data, scaler = preprocess_data(stock_data)
    model = load_model(model_path)

    # Prepare test data
    time_step = 60
    test_size = 0.2
    validation_size = 0.15
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + validation_size), shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + validation_size), shuffle=False)

    # Make predictions on test data
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate and display metrics
    mape = calculate_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MAPE: {mape:.2f}%")
    print(f"R-squared: {r2:.4f}")