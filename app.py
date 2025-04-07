import os
from flask import Flask, render_template, request, jsonify, session, redirect
from src.data_loader import fetch_stock_data, preprocess_data
from src.predict import predict_future_prices
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from flask_session import Session
import mysql.connector
import random
import string
import smtplib
from email.message import EmailMessage
import time, datetime
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import joblib  # For saving and loading the scaler

app = Flask(__name__)

MODEL_PATH = "models/enhanced_lstm_stock_model.h5"
SCALER_PATH = "models/scaler.pkl"  # Path to save/load the scaler

##### Establishing a connection with database
connection = mysql.connector.connect(
    host='127.0.0.1',
    database='stoxify_db',
    user='root',
    password='$9Gamb@098',
)

app.config["SECRET_KEY"] = "a76001519deeea4dde21a83b5f773301d3088d62536c4a1412c8b9d4184c807e"  # Change this in production
app.config["SESSION_TYPE"] = "filesystem"  # Store session data on the server
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
Session(app)

cursor = connection.cursor()

global ticker
ticker = None

global prediction
prediction = None

global current_day_price
current_day_price = None

global plot_data
plot_data = None

###### FUNCTION TO UPDATE THE STOCK PRICES IN THE TABLE ##########
def update_prices():
    cursor.execute("SELECT ticker_name FROM saved_stocks")
    saved_stocks = [row[0].upper() for row in cursor.fetchall()]

    if saved_stocks:
        # Fetch stock data for all tickers at once
        stock_data = yf.download(saved_stocks, period="1d", progress=False)["Close"].iloc[-1]

        # Construct a bulk update query
        update_values = []
        for ticker in saved_stocks:
            if ticker in stock_data:
                current_price = float(stock_data[ticker])
                update_values.append((current_price, ticker))

        # Execute the bulk update
        query = "UPDATE saved_stocks SET current_price = %s WHERE ticker_name = %s"
        cursor.executemany(query, update_values)
        connection.commit()
        print("Update successful")
    else:
        print("No stock data")

#### automate this update prices
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_prices, trigger="cron", hour=17, minute=0)  # Runs at 5:00 PM every day
scheduler.start()

# Shut down scheduler on app exit
atexit.register(lambda: scheduler.shutdown())

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form["ticker"]
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        # Fetch data for the selected stock
        stock_data = fetch_stock_data(ticker)

        if stock_data is None or stock_data.empty:
            return render_template("index.html",user_saved_stocks=get_user_stocks(),user_name=session.get('user_name'), stock_prediction="Error: Stock data not found.")

        # Extract last available date **before** transforming into NumPy array
        last_date = pd.to_datetime(stock_data.index[-1])

        # Preprocess data
        print("PreProcecessing the data ")
        scaled_data, scaler = preprocess_data(stock_data)
        current_day_price = float(stock_data['Close'].iloc[-1])

        # Extract last 60 days and ensure correct shape
        last_60_days = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60)
        last_60_days_scaled = scaler.transform(last_60_days.values)
        last_60_days_scaled = np.array(last_60_days_scaled).reshape(1, 60, 5)

        print("Shape of input data before prediction:", last_60_days_scaled.shape)  # Debugging

        
        # Pass `last_date` instead of `last_60_days.index[-1]`
        try:
            print("Predicting the future prices")
            predictions, date_range = predict_future_prices(MODEL_PATH, scaler, last_60_days, last_date, end_date)

            print("generating the prices")
            # Generate the plot
            plot_data = generate_plot(predictions, date_range)

            # Format the prediction message
            prediction = f"Predicted prices for {ticker} from {start_date} to {end_date} generated successfully."
        except ValueError as e:
            prediction = str(e)

        return render_template("index.html",stock_plot_data=plot_data,
                               user_saved_stocks=get_user_stocks(),
                               user_name=session.get('user_name'),
                               selected_ticker=ticker,
                               stock_prediction=prediction,
                               curr_day_price=current_day_price)

    return render_template('index.html')

######### FUNCTION TO RETURN TO MAIN ##########
@app.route("/home", methods=["GET"])
def home():
    return render_template("index.html",stock_plot_data=plot_data, user_saved_stocks=get_user_stocks(), prediction=None,user_name=session.get('user_name'),)

@app.route("/welcome")
def welcome():
    return render_template("welcome.html")

####### FUNCTION TO CHECK IF USERNAME ALREADY EXISTS ##########
def check_username_exists(user_name):
    cursor.execute("SELECT EXISTS(SELECT 1 FROM users WHERE user_name=%s)", (user_name,))
    return cursor.fetchone()[0] == 1

###### FUNCTION TO GET THE USERS SAVED STOCKS ###########
def get_user_stocks():
    if session.get('user_id'):
        cursor.execute("SELECT ticker_name, current_price FROM saved_stocks WHERE user_id=%s", (session['user_id'],))
        res = cursor.fetchall()
        if res:
            user_stocks = {ticker: price for ticker, price in res}
            return user_stocks
    return None

####### FUNCTION TO CHECK IF EMAIL ALREADY EXISTS ##########
def check_email_exists(email):
    cursor.execute("SELECT EXISTS(SELECT 1 FROM users WHERE email=%s)", (email,))
    return cursor.fetchone()[0] == 1

###### SEND MAIL ##########
def sendMail(email, gen_code):
    sender_email = "dum31555@gmail.com"
    sender_password = "dweg wzyz mbfa wvkv"

    msg = EmailMessage()
    msg.set_content(f"Your verification code is: {gen_code}\n\nThis code is valid for 5 minutes.")
    msg['Subject'] = "Email Verification Code"
    msg['From'] = sender_email
    msg['To'] = email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("Mail sent successfully")
        return True
    except Exception as e:
        print("Failed to send email:", e)
        return False

######### VERIFY EMAIL AND CREATE THE USER IN THE DB ########
@app.route('/verify_mail', methods=["POST"])
def verify_email():
    entered_otp = request.form.get('user_code')
    if entered_otp == gen_code:
        cursor.execute("INSERT INTO users (user_name, name, password, phone_no, email) VALUES (%s, %s, %s, %s, %s)",
                       (user_name, name, password, phone_no, email))
        connection.commit()
        return render_template('welcome.html', message="Profile creation successful😉👍🏿")
    else:
        return render_template('verify.html', message="Invalid OTP🧐")

########## FUNCTION TO REGISTER ###########
@app.route("/register", methods=["POST"])
def register():
    global user_name, name, password, phone_no, email
    user_name = request.form.get("username")
    name = request.form.get("name")
    password = request.form.get("password")
    phone_no = request.form.get("number")
    email = request.form.get("email")

    if check_username_exists(user_name):
        return render_template('welcome.html', message="This Username Already Exists, dude 🤦‍♂️", username=user_name, name=name, email=email, number=phone_no)

    if check_email_exists(email):
        return render_template('welcome.html', message="Yo This mail already exists 😒", username=user_name, name=name, email=email, number=phone_no)

    #### GENERATING A RANDOM CODE THAT CONTAINS UPPERCASE, LOWERCASE, AND DIGITS ##########
    global gen_code, timestamp
    gen_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    timestamp = time.time()  #### STORES THE CURRENT TIME
    if sendMail(email, gen_code):  # If email sending is successful
        return render_template('verify.html', message="OTP has been sent to your mail")
    else:
        return render_template('welcome.html', message="Failed To send the mail 😞", username=user_name, name=name, email=email, number=phone_no)

######## SIGN IN ########
@app.route('/signin', methods=["POST"])
def sign_in():
    user_name = request.form.get('user_name')
    pass_in = request.form.get('password')

    if check_username_exists(user_name):
        cursor.execute("SELECT password, user_id FROM users WHERE user_name=%s", (user_name,))
        user_data = cursor.fetchone()

        if user_data[0] == pass_in:
            session['user_name'] = user_name
            session['user_id'] = user_data[1]
            return render_template("index.html",stock_plot_data=plot_data, user_saved_stocks=get_user_stocks(), user_name=session.get('user_name'), selected_ticker=ticker, stock_prediction=prediction, curr_day_price=current_day_price)
        else:
            message = "Incorrect Password"
            return render_template('welcome.html', message=message)
    else:
        return render_template('welcome.html', message="User Does Not Exist")

def generate_plot(predictions, date_range):
    """
    Generate an interactive plot for the predicted prices over the selected date range.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=date_range,
        y=predictions,
        mode="lines+markers",
        marker=dict(size=6, color="purple"),
        line=dict(color="purple"),
        hoverinfo="x+y",
        name="Predicted Price"
    ))

    # Customize layout
    fig.update_layout(
        title="Predicted Stock Prices",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x",  # Show hover info only when mouse is near a point
        template="plotly_dark"  # Optional: Use a dark theme
    )

    return fig.to_html(full_html=False)

######### SAVE STOCK ######
@app.route("/saved_stock/<string:ticker_name>/<float:current_day_price>", methods=["GET"])
def save_stock(ticker_name, current_day_price):Ṇ
    cursor.execute("SELECT EXISTS(SELECT 1 FROM saved_stocks WHERE ticker_name=%s)", (ticker_name,))
    if cursor.fetchone()[0] == 1:
        return render_template("index.html",stock_plot_data=plot_data, user_saved_stocks=get_user_stocks(), user_name=session.get('user_name'), selected_ticker=ticker, stock_prediction=prediction, curr_day_price=current_day_price, message="Stock Already Saved 😉")
    else:
        cursor.execute("INSERT INTO saved_stocks (user_id, ticker_name, current_price) VALUES (%s, %s, %s)",
                       (session['user_id'], ticker_name, current_day_price))
        connection.commit()
        return render_template("index.html",stock_plot_data=plot_data, user_saved_stocks=get_user_stocks(), user_name=session.get('user_name'), selected_ticker=ticker, stock_prediction=prediction, curr_day_price=current_day_price, message="Stock's Now Under The Watch 🤓")

########## DELETE A STOCK FROM THE SAVED STOCKS ########
@app.route('/delete_stock/<string:ticker_name>', methods=["GET"])
def delete_stock(ticker_name):
    cursor.execute("DELETE FROM saved_stocks WHERE ticker_name=%s AND user_id=%s", (ticker_name, session['user_id']))
    connection.commit()
    return render_template("index.html", user_saved_stocks=get_user_stocks(), user_name=session.get('user_name'), selected_ticker=ticker, stock_prediction=prediction, stock_plot_data=plot_data, curr_day_price=current_day_price, message="Stock's out of the radar 🙌")

######### LOGOUT ########
@app.route('/logout')
def logout():
    session.clear()
    return render_template('index.html')

if __name__ == "__main__":
    app.run(port=1997, debug=True)