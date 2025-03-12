import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from data_loader import fetch_stock_data, preprocess_data

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def analyze_errors(y_true, y_pred, dates):
    """Analyzes misclassifications and finds largest prediction errors."""
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
    """Measures the time taken for predictions."""
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    avg_time_per_sample = elapsed_time / len(X_test)
    
    print(f"\nTotal Prediction Time: {elapsed_time:.4f} seconds")
    print(f"Average Time per Prediction: {avg_time_per_sample:.6f} seconds")

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

    # Calculate metrics
    mape = calculate_mape(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mean_error = np.mean(np.abs(y_true - y_pred))
    max_error = np.max(np.abs(y_true - y_pred))

    # Display results
    print(f"\nEvaluation Metrics:")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Error: {mean_error:.2f}")
    print(f"Max Error: {max_error:.2f}")

    # Analyze misclassifications
    analyze_errors(y_true, y_pred, dates=pd.date_range(start="2023-01-01", periods=len(y_true), freq='D'))

    # Measure prediction speed
    measure_prediction_time(model, X_test)
