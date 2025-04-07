import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(LSTM(units=32, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=1, activation='linear'))  # Linear activation for regression
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def prepare_training_data(scaled_data, time_step=90, test_size=0.2, validation_size=0.15):
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + validation_size), shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + validation_size), shuffle=False)

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    from data_loader import fetch_stock_data, preprocess_data
    
    ticker = "AMZN"
    stock_data = fetch_stock_data(ticker)
    scaled_data, scaler = preprocess_data(stock_data)
    
    time_step = 90  # Reduced to 90 days
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_training_data(scaled_data, time_step)
    
    model = create_lstm_model((X_train.shape[1], 1))
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-5, verbose=1)
    
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), 
              verbose=1, callbacks=[early_stopping, reduce_lr])
    
    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    
    # Save the trained model
    model.save("models/latest_lstm_stock_model.h5")
