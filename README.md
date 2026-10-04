# TCS Stock Price Prediction Engine

A time-series forecasting pipeline built in Python that predicts the next trading day's closing price of Tata Consultancy Services (`TCS.NS`). It fetches live historical market data dynamically via Yahoo Finance (`yfinance`) and implements Stochastic Gradient Descent (`SGDRegressor`) with strict scaling normalization to prevent data leakage.

## Key Features

- **Live Data Ingestion**: Dynamically fetches up-to-date market data via the `yfinance` API, ensuring the model always predicts based on the most recent trading session.
- **Custom NSE Calendar**: Utilizes Pandas `CustomBusinessDay` to accurately forecast dates by automatically skipping weekends and Indian stock market holidays (e.g., Gandhi Jayanti, Diwali).
- **Leakage-Free Preprocessing**: Features are normalized using `StandardScaler` fitted strictly on training data to prevent future look-ahead bias.
- **Chronological Splitting**: Enforces a strict 80/20 chronological train-test split, preserving time-series data integrity without random shuffling.
- **Baseline Evaluation**: Benchmarks prediction performance (MAE and $R^2$) against a Naive Baseline model (predicting $T+1$ price equals $T$ price) to prove actual learning.

## Project Architecture

1. **Target Feature Creation**: Creates the target variable by shifting the `Close` price by -1 day (`Next_Day_Close`).
2. **Feature Set**: Trains on core market indicators: `Open`, `High`, `Low`, `Close`, and `Volume`.
3. **Model Training**: Optimizes SGD weights over 200 epochs using constant learning rates and tracks validation loss.
