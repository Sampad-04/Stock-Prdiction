import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pickle

# Choose a representative Indian stock (e.g., Reliance)
ticker = "RELIANCE.NS"

df = yf.download(ticker, start="2020-01-01", end="2023-12-31")
df = df[['Close']].dropna()
df['Prev_Close'] = df['Close'].shift(1)
df.dropna(inplace=True)

X = df[['Prev_Close']]
y = df['Close']

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("✅ Model and scaler saved successfully.")
