import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from data_loader import fetch_stock_data, preprocess_data

def calculate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def analyze_errors(y_true, y_pred, dates):
    error_df = pd.DataFrame({
        'Date': dates,
        'Actual Price': y_true,
        'Predicted Price': y_pred,
        'Error': np.abs(y_true - y_pred)
    })
    error_df = error_df.sort_values(by='Error', ascending=False)
    print("\nTop 5 Largest Prediction Errors:")
    print(error_df.head(5))
    return error_df

def measure_prediction_time(model, X_test):
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()
    print(f"\nTotal Prediction Time: {end_time - start_time:.4f} seconds")
    print(f"Average Time per Prediction: {(end_time - start_time) / len(X_test):.6f} seconds")

if __name__ == "__main__":
    model_path = "models/enhanced_lstm_stock_model.h5"
    ticker = "AMZN"
    stock_data = fetch_stock_data(ticker)
    scaled_data, scaler = preprocess_data(stock_data)
    model = load_model(model_path)
    
    time_step = 60  # Ensure consistency with training
    test_size = 0.2
    validation_size = 0.15
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i])  # Keep all 5 features
        y.append(scaled_data[i, 0])  # Target is still just the first feature

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], time_step, 5)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + validation_size), shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(test_size + validation_size), shuffle=False)
    
    print("Model input shape:", model.input_shape)
    print("X_test shape before prediction:", X_test.shape)
    
    # Get predictions
    y_pred_scaled = model.predict(X_test)
    
    # Create a dummy array with 5 features for inverse transform
    dummy_array = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
    dummy_array[:, 0] = y_pred_scaled.flatten()  # Only set the first feature
    y_pred = scaler.inverse_transform(dummy_array)[:, 0]  # Inverse transform and take first column
    
    # Do the same for y_test
    dummy_array_test = np.zeros((len(y_test), scaled_data.shape[1]))
    dummy_array_test[:, 0] = y_test.flatten()
    y_true = scaler.inverse_transform(dummy_array_test)[:, 0]
    
    mape = calculate_mape(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nEvaluation Metrics:")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Get actual dates for the test period
    test_dates = stock_data.index[-len(y_test):]
    analyze_errors(y_true, y_pred, dates=test_dates)
    measure_prediction_time(model, X_test)