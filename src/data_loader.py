import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import yfinance as yf
from datetime import datetime

# Get the current date
current_date = datetime.today().strftime('%Y-%m-%d')

def fetch_stock_data(ticker):
    ticker = ticker.upper()  # Ensure uppercase ticker
    stock = yf.download(ticker, start="2021-01-01", end=current_date, auto_adjust=False)

    if stock.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    if 'Close' not in stock.columns:
        raise KeyError(f"'Close' column is missing in fetched data for {ticker}. Columns found: {stock.columns}")

    stock.ffill(inplace=True)
    return stock

def preprocess_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Ensure correct range
    scaled_data = scaler.fit_transform(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']])
    print(scaled_data)
    return scaled_data, scaler


if __name__ == "__main__":
    # Example usage
    ticker = "AMZN"
    stock_data = fetch_stock_data(ticker)
    scaled_data, scaler = preprocess_data(stock_data)
    #print(scaled_data.head())