import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define constants
stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
start_date = "2019-01-01"
end_date = "2025-03-18"
lookback = 60  # No of days to look back

# Fetch stock data
df = yf.download(stocks, start=start_date, end=end_date, interval="1d", auto_adjust=False)['Close']
df.ffill(inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=stocks, index=df.index)

# Convert to time-series format
X, Y = [], []
for i in range(lookback, len(df_scaled)):
    X.append(df_scaled.iloc[i-lookback:i].values)
    Y.append(df_scaled.iloc[i, 0])

X, Y = np.array(X), np.array(Y)

# Split data
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.2)
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
Y_train, Y_val, Y_test = Y[:train_size], Y[train_size:train_size+val_size], Y[train_size+val_size:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, len(stocks))),
    Dropout(0.198),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss="mean_squared_error")

# Train the model
history = model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_val, Y_val))

# Evaluate the model
test_loss = model.evaluate(X_test, Y_test)
print(f'Test Loss: {test_loss}')

# Save the trained model
model.save("models/enhanced_lstm_stock_model.h5")
print("✅ Model training complete and saved as 'models/enhanced_lstm_stock_model.h5'")

# Generate and save the Training & Validation Loss Graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label="Training Loss", color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label="Validation Loss", color='red', linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Graph")
plt.legend()
plt.grid(True)

# Save the graph automatically
graph_path = "models/enhanced_model_training_validation_loss.png"
plt.savefig(graph_path)
plt.show()

print(f"📊 Training Loss Graph saved as '{graph_path}'")
