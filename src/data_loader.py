import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import yfinance as yf

def fetch_stock_data(ticker):
    ticker = ticker.upper()  # Ensure uppercase ticker
    stock_data = yf.download(ticker, period="max", auto_adjust=False, progress=False)

    #print(stock_data.head())  # Check the first few rows
    #print(stock_data.columns)  # Print available columns

    if stock_data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    if 'Close' not in stock_data.columns:
        raise KeyError(f"'Close' column is missing in fetched data for {ticker}. Columns found: {stock_data.columns}")

    return stock_data[['Close']]




def preprocess_data(stock_data):
    """
    Preprocess data for training.
    """
    # Use 'Close' price for prediction and ensure it's a DataFrame
    data = stock_data[['Close']].copy()  # Use copy to avoid SettingWithCopyWarning

    # Normalize data (scaling between 0 and 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))  # Fit and transform on 'Close' column

    return data.values, scaler  # Return the DataFrame with scaled 'Close' prices

if __name__ == "__main__":
    # Example usage
    ticker = "AMZN"
    stock_data = fetch_stock_data(ticker)
    scaled_data, scaler = preprocess_data(stock_data)
    #print(scaled_data.head())