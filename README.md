# 📈 STOXIFY: Deep Learning Based Stock Prediction System

Stoxify is an LSTM based stock prediction system built to help forecast stock trends and prices. Designed for personal investors, advisors, and finance enthusiasts, Stoxify provides real-time predictions with trend analysis, and visualization
---
## 🚀 Features

- 🔍 Real-Time Stock Data via `yfinance`
- 🧠 Deep Learning Prediction Models (LSTM)
- 📊Trend Analysis
- 📧 Email Verification & Alerts
- 🗓️ Automated Background Updates (APScheduler)
- 💾 MySQL Database Integration for user and stock history
- 📈 Interactive Visualizations using Plotly
---
## 🛠️ Tech Stack

| Layer       | Technologies Used                         |
|-------------|-------------------------------------------|
| Backend     | Python, Flask, LSTM (TensorFlow/Keras)    |
| Frontend    | HTML, CSS, Bootstrap                      |
| Database    | MySQL                                     |
| Visualization | Plotly                                 |
| Scheduling  | APScheduler                               |
| Data Source | yFinance                                  |

---

## 📂 Project Structure

stoxify/ │ ├── models/ # Trained LSTM models and training artifacts │ ├── enhanced_lstm_stock_model.h5 │ ├── enhanced_model_training_val_loss.png │ └── updated_lstm_stock_model.h5 │ ├── src/ # Source code for data loading, training, prediction │ ├── data_loader.py │ ├── evaluate.py │ ├── generate_train_loss.py │ ├── lstm_archi_generator.py │ ├── model.py │ ├── new_model.py │ └── predict.py │ ├── static/ # Static assets (CSS, JS, Images, Videos) │ ├── bg_img.png │ ├── first.css │ ├── loading2.jpg │ ├── loading3.mp4 │ ├── loading4.mp4 │ ├── scripts.js │ ├── style.css │ └── welcome_bg.png │ ├── templates/ # HTML templates for Flask frontend │ ├── index.html │ ├── verify.html │ └── welcome.html │ ├── app.py # Flask application entry point ├── requirements.txt # Python dependencies ├── training_validation_loss.png # Training/Validation loss plot ├── .gitignore # Git ignored files └── README.md # Project documentation (you’re reading it!)
---
🙌 Acknowledgements
•	yFinance
•	Keras
•	Flask
•	Plotly, APScheduler, and all open-source libraries used

⚠️ Disclaimer: Stoxify is a personal project and is for educational and research purposes only. Not financial advice. Always Do Your Own Research (DYOR).
