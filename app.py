from flask import Flask, render_template, request
from src.data_loader import fetch_stock_data, preprocess_data
from src.predict import predict_future_price
from src.model import create_lstm_model, prepare_training_data
import os

app = Flask(__name__)

# Load the trained model (or train if not already trained)
MODEL_PATH = "models/lstm_stock_model.h5"
if not os.path.exists(MODEL_PATH):
    print("Training the model...")
    stock_data = fetch_stock_data("AAPL")  # Default ticker for training
    scaled_data, scaler = preprocess_data(stock_data)
    X, y = prepare_training_data(scaled_data)
    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)
    model.save(MODEL_PATH)
    print("Model trained and saved.")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        ticker = request.form["ticker"]
        future_date = request.form["future_date"]

        # Fetch data and make prediction
        stock_data = fetch_stock_data(ticker)
        scaled_data, scaler = preprocess_data(stock_data)
        last_60_days = stock_data['Close'][-60:]
        prediction = predict_future_price(MODEL_PATH, scaler, last_60_days, future_date)
        prediction = f"Predicted price for {future_date}: ${prediction:.2f}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(port=1997, debug=True)
