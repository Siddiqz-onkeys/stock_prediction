from src.data_loader import fetch_stock_data, preprocess_data

data = fetch_stock_data("AAPL","2019-01-01",end="2025-03-19")
scaled_data, scaler = preprocess_data(data)

print(data.tail())  # Check if data is fetched properly
print(scaled_data[:5])  # Check scaled values
