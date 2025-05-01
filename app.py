from flask import Flask, render_template, request
import yfinance as yf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        ticker = request.form["ticker"].upper()
        if not ticker.endswith(".NS"):
            ticker += ".NS"

        data = yf.download(ticker, period="5d")
        if data.empty:
            return render_template("index.html", prediction_text="❌ Invalid ticker or no data available.")

        latest_close = float(data["Close"].iloc[-1])  # Get the latest closing price
        latest_close_2d = np.array([[latest_close]])  # Convert to 2D array
        scaled = scaler.transform(latest_close_2d)    # Scale it
        prediction = float(model.predict(scaled)[0])  # Predict and convert to float

        result = f"📊 Predicted next closing price for {ticker}: ₹{prediction:.2f}"
        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error: {str(e)}")



@app.route("/plot", methods=["POST"])
def plot():
    ticker = request.form["ticker_plot"].upper()
    if not ticker.endswith(".NS"):
        ticker += ".NS"

    df = yf.download(ticker, period="90d")
    df = df[['Close']].dropna()
    df['Prev_Close'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    X = scaler.transform(df[['Prev_Close']])
    y = df['Close']
    y_pred = model.predict(X)

    plt.figure(figsize=(10, 5))
    plt.plot(y.index, y, label="Actual", linewidth=2)
    plt.plot(y.index, y_pred, label="Predicted", linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("₹ Price")
    plt.title(f"{ticker} - Actual vs Predicted Prices")
    plt.legend()
    plt.tight_layout()

    if not os.path.exists("static"):
        os.makedirs("static")
    plt.savefig("static/plot.png")
    plt.close()

    return render_template("plot.html", plot_url="static/plot.png")


if __name__ == "__main__":
    app.run(debug=True)