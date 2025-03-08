import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def predict_future_price(model_path, scaler, last_60_days, future_date):
    """
    Predict stock price for a specific future date.
    """
    # Load the trained model
    model = load_model(model_path)

    # Convert future date to datetime
    future_date = pd.to_datetime(future_date)
    current_date = pd.to_datetime(last_60_days.index[-1])  # Last date in historical data
    days_ahead = (future_date - current_date).days  # Number of days to predict

    if days_ahead < 1:
        raise ValueError("Future date must be after the current date.")

    # Predict step-by-step until the future date
    input_data = last_60_days[-60:].values.reshape(-1, 1)  # Ensure input_data is 2D
    predictions = []
    for _ in range(days_ahead):
        # Reshape input_data to 2D before scaling
        input_data_scaled = scaler.transform(input_data.reshape(-1, 1))
        input_data_scaled = input_data_scaled.reshape(1, -1, 1)  # Reshape for LSTM input
        predicted_price = model.predict(input_data_scaled)
        predictions.append(predicted_price[0, 0])
        # Update input_data with the predicted price
        input_data = np.append(input_data[1:], predicted_price).reshape(-1, 1)  # Ensure 2D

    # Inverse transform the predictions to get actual prices
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions[-1][0]  # Return the price for the specific future date

if __name__ == "__main__":
    # Example usage
    from data_loader import fetch_stock_data, preprocess_data

    ticker = "AAPL"
    future_date = "2024-01-15"  # Specific future date
    stock_data = fetch_stock_data(ticker)
    scaled_data, scaler = preprocess_data(stock_data)
    last_60_days = stock_data['Close'][-60:]  # Last 60 days of historical data
    predicted_price = predict_future_price("models/lstm_stock_model.h5", scaler, last_60_days, future_date)
    print(f"Predicted price for {future_date}: ${predicted_price:.2f}")
