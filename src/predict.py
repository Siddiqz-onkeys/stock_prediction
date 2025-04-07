import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def predict_future_prices(model_path, scaler, last_60_days, start_date, end_date):
    # Load the model
    model = load_model(model_path)
    
    # Convert dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    current_date = pd.to_datetime(last_60_days.index[-1])

    # Validate dates
    if start_date < current_date:
        raise ValueError("Start date must be after the current date.")
    if end_date <= start_date:
        raise ValueError("End date must be after start date.")

    total_days = (end_date - start_date).days
    date_range = pd.date_range(start=start_date, periods=total_days, freq='D')

    # Prepare ALL features (5 columns)
    input_data = last_60_days[['Open', 'High', 'Low', 'Close', 'Volume']].values
    input_data_scaled = scaler.transform(input_data)

    # Use rolling predictions
    predicted_prices = []
    rolling_window = input_data_scaled[-60:].tolist()  # Last 60 days of scaled data

    for _ in range(total_days):
        # Reshape for LSTM input (1, 60, 5)
        input_array = np.array(rolling_window).reshape(1, 60, 5)
        
        # Predict (output is just Close price)
        predicted_scaled = model.predict(input_array)[0, 0]  # Single value
        
        # Inverse transform the Close price
        dummy_array = np.zeros((1, 5))  # Match scaler's expected shape
        dummy_array[0, 3] = predicted_scaled  # Place at Close position (index 3)
        predicted_close_actual = scaler.inverse_transform(dummy_array)[0, 3]
        
        predicted_prices.append(predicted_close_actual)

        # Update rolling window: append predicted Close, reuse other features
        new_row = rolling_window[-1].copy()  # Copy last row
        new_row[3] = predicted_scaled  # Update Close price
        rolling_window.pop(0)  # Remove oldest
        rolling_window.append(new_row)  # Add new prediction

    return predicted_prices, date_range