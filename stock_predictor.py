# ==============================================================================
# PROJECT 2: STOCK PRICE PREDICTOR USING LINEAR REGRESSION
# AI Internship Project
# ==============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("="*60)
print("     📈 INITIALIZING AI STOCK PRICE PREDICTOR SYSTEM 📈     ")
print("="*60)

ticker = "AAPL"
print(f"\n⏳ Fetching live historical data for {ticker} from Yahoo Finance...")
data = yf.download(ticker, start="2024-01-01", end="2025-01-01")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

df = data[['Open', 'High', 'Low', 'Close']].dropna()
X = df[['Open', 'High', 'Low']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print("🎯 Model Training Status: SUCCESSFUL!")

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"📊 Model Evaluation - Mean Squared Error (MSE): {mse:.4f}")

print("\n📊 Generating Visual Performance Chart...")
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:50], label="Actual Closing Price", color='royalblue', linewidth=2, marker='o')
plt.plot(predictions[:50], label="AI Predicted Price", color='crimson', linestyle='--', linewidth=2, marker='x')

plt.title(f"{ticker} Stock Price Prediction - Actual vs AI Predicted", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Days (Sample Data)", fontsize=12)
plt.ylabel("Stock Price (USD $)", fontsize=12)
plt.legend(fontsize=11, loc="upper left")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig('stock_prediction_output.png', dpi=300)
print("\n" + "="*60)
print("🎉 SUCCESS: 'stock_prediction_output.png' saved successfully!")
print("✨ STOCK PREDICTOR COMPLETED SUCCESSFULLY ✨")
print("="*60)
