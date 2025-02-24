import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape):
    """
    Create an LSTM model for stock price prediction.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_training_data(scaled_data, time_step=60):
    """
    Prepare training data for LSTM.
    """
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

if __name__ == "__main__":
    # Example usage
    from data_loader import fetch_stock_data, preprocess_data

    ticker = "AAPL"
    stock_data = fetch_stock_data(ticker)
    scaled_data, scaler = preprocess_data(stock_data)
    time_step = 60
    X, y = prepare_training_data(scaled_data, time_step)
    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)
    model.save("models/lstm_stock_model.h5")
