import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data(ticker):
    """
    Fetch historical stock data up to the current date.
    """
    stock_data = yf.download(ticker, period="max")  # Fetch all available data
    return stock_data

def preprocess_data(stock_data):
    """
    Preprocess data for training.
    """
    # Use 'Close' price for prediction
    data = stock_data[['Close']].values

    # Normalize data (scaling between 0 and 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    stock_data = fetch_stock_data(ticker)
    scaled_data, scaler = preprocess_data(stock_data)
    print(scaled_data[:5])
