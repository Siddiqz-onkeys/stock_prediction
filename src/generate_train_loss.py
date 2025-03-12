import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from data_loader import fetch_stock_data, preprocess_data
from model import create_lstm_model

# Load stock data
ticker = "AMZN"
stock_data = fetch_stock_data(ticker)
scaled_data, scaler = preprocess_data(stock_data)

# Prepare data for LSTM
time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.35, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# ✅ Fix: Reinitialize the optimizer after loading the model
try:
    model = load_model("models/updated_lstm_stock_model.h5", compile=False)  # Load without compiling
    model.compile(optimizer="adam", loss="mean_squared_error")  # Reinitialize optimizer
    print("✅ Pre-trained model loaded and recompiled successfully.")
except:
    print("⚠️ Model not found, creating a new one...")
    model = create_lstm_model((X_train.shape[1], 1))

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model and capture history
history = model.fit(
    X_train, y_train, epochs=100, batch_size=32,
    validation_data=(X_val, y_val), verbose=1, callbacks=[early_stopping]
)

# Plot Training and Validation Loss Graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label="Training Loss", color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label="Validation Loss", color='red', linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Graph")
plt.legend()
plt.grid(True)
plt.savefig("training_validation_loss.png")  # Save the graph
plt.show()

print("✅ Training Loss Graph saved as 'training_validation_loss.png'")

