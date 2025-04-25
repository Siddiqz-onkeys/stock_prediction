import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Define constants
stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]  # Using 5 stocks
target_stock = "AAPL"  # Train on just one stock
start_date = "2019-01-01"
end_date = "2025-03-18"
lookback = 60

# Fetch stock data
df = yf.download(stocks, start=start_date, end=end_date, interval="1d", auto_adjust=False)['Close']
df.ffill(inplace=True)  # Forward-fill missing values

# Select only target stock (AAPL) to match model input
df = df[[target_stock]]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df.values.reshape(-1, 1))

# Convert to time-series format
X, Y = [], []
for i in range(lookback, len(df_scaled)):
    X.append(df_scaled[i-lookback:i])  # 60 time steps
    Y.append(df_scaled[i, 0])  # Predicting the next day's price

X, Y = np.array(X), np.array(Y)

# Reshape X to (samples, time_steps, features=1)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # 1 feature to match model

# Load pre-trained model
model = load_model("models/enhanced_lstm_stock_model.h5")

# Train the model again to get the loss history
history = model.fit(X, Y, epochs=50, batch_size=64, validation_split=0.2, verbose=1)

# Plot Training and Validation Loss Graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label="Training Loss", color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label="Validation Loss", color='red', linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Graph")
plt.legend()
plt.grid(True)
plt.savefig("enhanced_model_training_validation_loss.png")  # Save the graph
plt.show()

print("✅ Training Loss Graph saved as 'enhanced_model_training_validation_loss.png'")
