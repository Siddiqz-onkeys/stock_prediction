import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def  predict_future_prices(model_path, scaler, last_60_days, start_date, end_date):
    """
    Predict stock prices for the selected date range using a rolling window approach.
    """
    model = load_model(model_path)
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    current_date = pd.to_datetime(last_60_days.index[-1])

    if start_date <= current_date:
        raise ValueError("Start date must be after the current date.")
    if end_date <= start_date:
        raise ValueError("End date must be after start date.")

    total_days = (end_date - start_date).days
    date_range = pd.date_range(start=start_date, periods=total_days, freq='D')

    # Prepare input data
    input_data = last_60_days.values.reshape(-1, 1)
    input_data_scaled = scaler.transform(input_data)

    # Use rolling predictions
    predicted_prices = []
    rolling_window = input_data_scaled[-60:].tolist()  # Convert to list for easy updating

    for _ in range(total_days):
        input_array = np.array(rolling_window).reshape(1, 60, 1)  # Reshape for model input
        predicted_scaled = model.predict(input_array)[0, 0]  # Predict next day
        predicted_actual = scaler.inverse_transform([[predicted_scaled]])[0, 0]  # Convert back
        predicted_prices.append(predicted_actual)
        rolling_window.pop(0)  # Remove oldest entry
        rolling_window.append([predicted_scaled])  # Append new predicted value

    return predicted_prices, date_range
