import os
from flask import Flask, render_template, request,jsonify,session,redirect
from src.data_loader import fetch_stock_data, preprocess_data
from src.predict import predict_future_prices
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from flask_session import Session
import mysql.connector
import random
import string
import smtplib
from email.message import EmailMessage
import time,datetime
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
import atexit


app = Flask(__name__)

MODEL_PATH = "models/updated_lstm_stock_model.h5"

##### Establishing a connection with database
connection=mysql.connector.connect(
    host='stoxify.cvs82c00gy0z.ap-south-1.rds.amazonaws.com',
    database='stoxify-25',
    user='root',
    password='#plaSticr&25--',
    connection_timeout=300
)


app.config["SECRET_KEY"] = "a76001519deeea4dde21a83b5f773301d3088d62536c4a1412c8b9d4184c807e"  # Change this in production
app.config["SESSION_TYPE"] = "filesystem"  # Store session data on the server
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
Session(app)

cursor=connection.cursor()

global ticker
ticker=None

global prediction
prediction = None

global plot_data
plot_data = None

global current_day_price
current_day_price=None


###### FUNTION TO UPDATE THE STOCK PRICES IN THE TABLE ##########
def update_prices():
    cursor.execute("SELECT ticker_name FROM saved_stocks")
    saved_stocks = [row[0].upper() for row in cursor.fetchall()] 

    if saved_stocks:
        # Fetch stock data for all tickers at once
        stock_data = yf.download(saved_stocks, period="1d", progress=False)["Close"].iloc[-1]
        #print(stock_data)

        # Construct a bulk update query
        update_values = []
        for ticker in saved_stocks:
            #print(ticker)
            if ticker in stock_data:
                current_price = float(stock_data[ticker])
                print(current_price)
                update_values.append((current_price, ticker))
                

        # Execute the bulk update
        query = "UPDATE saved_stocks SET current_price = %s WHERE ticker_name = %s"
        cursor.executemany(query, update_values)
        connection.commit()
        print("Update successful")
        

        
    else:
        print("Not stock data")


#### automate this update prices
scheduler = BackgroundScheduler()
scheduler.add_job (func=update_prices, trigger="cron", hour=17 ,minute=0 ) # Runs at 5:00 PM every day
scheduler.start()


# Shut down scheduler on app exit
atexit.register(lambda: scheduler.shutdown())



######## THE INDEX FUNCTION ########
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        
        ticker = request.form["ticker"]

        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        # Fetch data for the selected stock
        stock_data = fetch_stock_data(ticker)
        scaled_data, scaler = preprocess_data(stock_data)
        current_day_price=float(stock_data['Close'].iloc[-1])
        last_60_days = stock_data['Close'][-60:]
        # print(stock_data.columns)  # Check column names
        # print(stock_data.head())   # Print first few rows


        # Make predictions
        try:
            predictions, date_range = predict_future_prices(MODEL_PATH, scaler, last_60_days, start_date, end_date)
            # print("so",predictions)
            
            # Generate the plot
            plot_data = generate_plot(predictions, date_range)
            
            # Format the prediction message
            prediction = f"Predicted prices for {ticker} from {start_date} to {end_date} generated successfully."
        except ValueError as e:
            prediction = str(e)
        return render_template("index.html",user_saved_stocks=get_user_stocks(),user_name=session.get('user_name'),selected_ticker=ticker, stock_prediction=prediction, stock_plot_data=plot_data,curr_day_price=current_day_price)
    return render_template('index.html')

######### FUNTION TO RETURN TO MAIN ##########
@app.route("/home",methods=["GET"])
def home():
    return render_template("index.html",user_saved_stocks=get_user_stocks(), prediction=None, plot_data=None)

@app.route("/welcome")
def welcome():
    return render_template("welcome.html")

####### FUNCTION TO CHECK IF USERNAME ALREADY EXISTS ##########
def check_username_exists(user_name):
    cursor.execute("SELECT EXISTS(SELECT 1 FROM users WHERE user_name=%s)",(user_name,))
    if cursor.fetchone()[0]==1:
        
        return True
    else:
        
        return False

###### FUNCTION TO GET THE USERS DSAVED STOCKS ###########
def get_user_stocks():
    print(session['user_id'])
    cursor.execute("SELECT ticker_name,current_price FROM saved_stocks WheRE user_id=%s",(session['user_id'],))
    res=cursor.fetchall()
    print(res)
    cursor.close()
    connection.close()
    if res:
        user_stocks={}
        for stock in res:
            ticker,price=stock
            user_stocks[ticker]=price              
        return user_stocks
    return None

    
####### FUNCTION TO CHECK IF USERNAME ALREADY EXISTS ##########
def check_email_exists(email):
    cursor.execute("SELECT EXISTS(SELECT 1 FROM users WHERE email=%s)",(email,))
    if cursor.fetchone()[0]==1:
        
        return True
    else:
        
        return False

###### SEND MAIL ##########
def sendMail(email,gen_code):
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
        print("mail sent successful;ly")
        return True
    except Exception as e:
        print("Failed to send email:", e)
        return False
           
######### VERIFY EMAIL AND CREATE THE USER IN THE DB ########
@app.route('/verify_mail',methods=["POST"])
def verify_email():
    
    entered_otp=request.form.get('user_code')
    if entered_otp==gen_code:
        cursor.execute("INSERT INTO users (user_name,name,password,phone_no,email) VALUES (%s,%s,%s,%s,%s)",(user_name,name,password,phone_no,email,))
        connection.commit()
        
        user_cr=True
        return render_template('welcome.html',message="Profile creation successful😉👍🏿")
    else:
        return render_template('verify.html',message="Invalid OTP🧐")


########## FUNTION TO REGISTER ###########
@app.route("/register",methods=["POST"])
def register():
    global user_name,name,password,phone_no,email
    user_name=request.form.get("username")
    name=request.form.get("name")
    password=request.form.get("password")
    phone_no=request.form.get("number")
    email=request.form.get("email")
    
    if check_username_exists(user_name):
            render_template('welcome.html',name=name,phone_no=phone_no,email=email,username=user_name)
            return render_template('welcome.html',message="This Username Already Exists, dude 🤦‍♂️",username=user_name,name=name,email=email,number=phone_no)
        
        
    if check_email_exists(email):
        render_template('welcome.html',name=name,phone_no=phone_no,email=email,username=user_name)
        return render_template('welcome.html',message="Yo This mail already exists 😒",username=user_name,name=name,email=email,number=phone_no)   
    
    #### GENERATING A RANDOM CODE THAT CONTAINS UPPERCASE,LOWERCASE,AND DIGITS ##########
    global gen_code,timestamp
    gen_code=''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    timestamp = time.time()  #### STORES THE CURRENT TIME
    #return render_template('verify.html')
    if sendMail(email, gen_code):  # If email sending is successful
        return render_template('verify.html',message="OTP has been sent to your mail")
    else:
        return render_template('welcome.html',message="Failed To send the mail 😞",username=user_name,name=name,email=email,number=phone_no)

######## SIGN IN ########
@app.route('/signin',methods=["POST"])
def sign_in():
    
    user_name=request.form.get('user_name')
    pass_in=request.form.get('password')
    
    if check_username_exists(user_name):
        cursor.execute("SELECT password,user_id FROM users WHERE user_name=%s",(user_name,))
        user_data=cursor.fetchone()
        
        if user_data[0]==pass_in:
            session['user_name']=user_name
            session['user_id']=user_data[1]
            
            
            return render_template("index.html",user_saved_stocks=get_user_stocks(),user_name=session.get('user_name'),selected_ticker=ticker, stock_prediction=prediction, stock_plot_data=plot_data,curr_day_price=current_day_price)
        else:
            message="Incorrect Password"
            return render_template('welcome.html',message=message)
    else:
        return render_template('welcome.html',message=" User Does Not Exist")


######## FUNCTINO TO PLOT THE DATA ###########
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
        xaxis=dict(showticklabels=False),  # Remove x-axis labels
        yaxis_title="Price ($)",
        hovermode="x",  # Show hover info only when mouse is near a point
        template="plotly_dark"  # Optional: Use a dark theme
    )

    return fig.to_html(full_html=False)

######### SAVE STOCK ######
@app.route("/saved_stock/<string:ticker_name>/<float:current_day_price>",methods=["GET"])
def save_stock(ticker_name,current_day_price):
    
    cursor.execute("SELECT EXISTS(SELECT 1 FROM saved_stocks WHERE ticker_name=%s)",(ticker_name,))
    if cursor.fetchone()[0]==1:
        
        return render_template("index.html",user_saved_stocks=get_user_stocks(),user_name=session.get('user_name'),selected_ticker=ticker, stock_prediction=prediction, stock_plot_data=plot_data,curr_day_price=current_day_price,message="Stock Already Saved 😉")
    else:
        cursor.execute("INSERT INTO saved_stocks (user_id,ticker_name,current_price) VALUES (%s,%s,%s)",(session['user_id'],ticker_name,current_day_price,))
        connection.commit()
        
        return render_template("index.html",user_saved_stocks=get_user_stocks(),user_name=session.get('user_name'),selected_ticker=ticker, stock_prediction=prediction, stock_plot_data=plot_data,curr_day_price=current_day_price,message="Stock's Now Under The Watch 🤓")

######### LOGOUT ########
@app.route('/logout')
def logout():
    session.clear()
    return render_template('index.html')


if __name__ == "__main__":
    app.run(port=1997, debug=True)
