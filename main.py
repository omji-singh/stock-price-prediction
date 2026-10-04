from pandas.tseries.offsets import CustomBusinessDay
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
#      DATA INGESTION & SETUP (LIVE)
# ==========================================
TICKER = 'TCS.NS'
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
EPOCHS = 200

print(f"Fetching latest live market data for {TICKER}...")
# Fetch data up to today using period="5y" instead of fixed start/end dates
stock_data = yf.download(TICKER, period="5y", auto_adjust=False)

if stock_data.empty:
    raise ValueError("No data downloaded. Check ticker symbol.")

if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.droplevel(1)

# ==========================================
#    FEATURE ENGINEERING & TODAY'S ROW
# ==========================================
# Shift Close price up by -1 to set tomorrow's price as the target
stock_data['Next_Day_Close'] = stock_data['Close'].shift(-1)

# 1. Save TODAY'S live feature row BEFORE dropping NaNs
today_features = stock_data[FEATURES].iloc[-1:]
today_date = today_features.index[0].strftime('%Y-%m-%d')

# 2. Remove the trailing NaN row from the training/testing dataset
labeled_data = stock_data[FEATURES + ['Next_Day_Close']].dropna()

X = labeled_data[FEATURES]
y = labeled_data['Next_Day_Close']

# Chronological split (80% train, 20% test)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ==========================================
#     SCALING (PREVENTING DATA LEAKAGE)
# ==========================================
# Fit scalers ONLY on training data to prevent the model from "peeking" at the future
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))

X_train_scaled = x_scaler.transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
# ==========================================
#      MODEL TRAINING (SGD REGRESSOR)
# ==========================================
model = SGDRegressor(learning_rate='constant', eta0=0.01, penalty=None, random_state=42)

train_losses, val_losses = [], []
rng = np.random.default_rng(42)

print("Starting training loop...")
for epoch in range(EPOCHS):
    # Shuffle training data internally for stochastic gradient descent
    idx = rng.permutation(len(X_train_scaled))
    model.partial_fit(X_train_scaled[idx], y_train_scaled[idx])

    # Record Mean Squared Error (MSE) for learning curves
    train_losses.append(np.mean((model.predict(X_train_scaled) - y_train_scaled) ** 2))
    val_losses.append(np.mean((model.predict(X_test_scaled) - y_test_scaled) ** 2))

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train MSE: {train_losses[-1]:.5f} | Val MSE: {val_losses[-1]:.5f}")

# Save the trained model and scalers
bundle = {'model': model, 'x_scaler': x_scaler, 'y_scaler': y_scaler, 'features': FEATURES}
joblib.dump(bundle, 'stock_model.joblib')
print("\nModel pipeline saved to 'stock_model.joblib'.\n")

# ==========================================
#     EVALUATION, FORECASTING & PLOTS
# ==========================================
def predict_price(frame):
    """Helper function to scale input features, predict, and inverse transform back to INR."""
    scaled_input = model.predict(x_scaler.transform(frame[FEATURES]))
    return y_scaler.inverse_transform(scaled_input.reshape(-1, 1)).ravel()

# ------------------------------------------
# 1. TEST SET EVALUATION vs NAIVE BASELINE
# ------------------------------------------
y_pred = predict_price(X_test)
naive_predictions = X_test['Close'].values

print("--- MODEL PERFORMANCE METRICS ---")
print(f"Model MAE      : {mean_absolute_error(y_test, y_pred):.2f} INR")
print(f"Naive MAE      : {mean_absolute_error(y_test, naive_predictions):.2f} INR")
print(f"R-Squared (R2) : {r2_score(y_test, y_pred):.4f}")

# ------------------------------------------
# 2. LIVE MARKET FORECAST (LAST vs NEXT DAY)
# ------------------------------------------
# Get the exact date and closing price of the last known trading session
last_date_obj = today_features.index[0]
last_date_str = last_date_obj.strftime('%Y-%m-%d')
last_actual_close = today_features['Close'].values[0]

# Define NSE Market Holidays (e.g., Gandhi Jayanti, Dussehra, Diwali)
nse_holidays = ['2026-10-02', '2026-10-20', '2026-11-10'] 

# Create a custom trading calendar that skips weekends AND market holidays
TradingDay = CustomBusinessDay(holidays=nse_holidays)

# Calculate the next actual trading session
next_date_obj = last_date_obj + TradingDay
next_date_str = next_date_obj.strftime('%Y-%m-%d')

# Predict closing price for the next trading session
next_day_prediction = predict_price(today_features)[0]

print("\n" + "="*50)
print(" 📈 LIVE MARKET FORECAST: TCS.NS")
print("="*50)
print(f"LAST TRADING SESSION : {last_date_str}")
print(f"ACTUAL CLOSING PRICE : {last_actual_close:.2f} INR")
print("-" * 50)
print(f"NEXT TRADING SESSION : {next_date_str}")
print(f"PREDICTED CLOSE PRICE: {next_day_prediction:.2f} INR")
print("="*50 + "\n")

# ------------------------------------------
# 3. VISUALIZATIONS
# ------------------------------------------
# Plot 1: Model Training Curve (Train vs Validation MSE)
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train MSE')
plt.plot(val_losses, label='Validation MSE')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE (scaled)')
plt.title('Model Training Curve (SGD Regressor)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Historical Predictions vs Actual Close Prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Next Day Close', color='blue', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted Next Day Close', color='red', linestyle='--', linewidth=2)
plt.title(f'{TICKER} Stock Price Prediction vs Actual (Test Set)')
plt.xlabel('Date')
plt.ylabel('Closing Price (INR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
