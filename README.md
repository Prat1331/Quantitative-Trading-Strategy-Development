📈 Quantitative Trading Strategy Development
Regime-Aware Intraday Market Prediction using Machine Learning
📌 Project Overview

This project focuses on building a regime-aware intraday trading prediction system using classical machine learning models and financial feature engineering.
The system predicts the next-bar market direction (UP / DOWN) based on price action, volatility, and detected market regimes.

The project includes:

Feature engineering using technical indicators
Market regime detection (trend × volatility)
Supervised ML models (Logistic Regression & XGBoost)
A Streamlit web application for interactive prediction
Model persistence and reproducibility

🎯 Problem Statement

Financial markets behave differently under different conditions (e.g., trending vs ranging, high vs low volatility).
A single predictive model without regime awareness often performs poorly.

Goal:

Build a regime-aware ML pipeline that adapts predictions based on current market conditions.

🧠 Key Concepts Used

Log returns
Exponential Moving Averages (EMA)
Rolling volatility
Trend & volatility regime classification
One-hot encoding of regimes
Time-series aware train/test split
Probabilistic classification
Model deployment with Streamlit

🏗️ Project Structure
Quantitative-Trading-Strategy-Development/
│
├── app.py                  # Streamlit application
├── train.py                # Model training pipeline
├── features.py             # Feature engineering logic
├── config.py               # (Optional) config/constants
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
│
├── data/
│   └── spot_cleaned.csv    # Cleaned OHLCV dataset
│
├── models/
│   ├── logistic_model.pkl  # Trained Logistic Regression model
│   ├── xgb_model.pkl       # Trained XGBoost model
│   └── final_features.pkl  # Exact feature list used in training

⚙️ Feature Engineering

All feature engineering is centralized in features.py to ensure training and inference consistency.

Engineered Features
log_return – Logarithmic price returns
ema – Exponential Moving Average
ema_slope – Trend strength
rolling_vol – Rolling volatility
trend_regime – UPTREND / DOWNTREND
vol_regime – HIGH_VOL / LOW_VOL
market_regime – Combined regime (one-hot encoded)

🧪 Target Variable

Binary classification

target = 1 → next bar return > 0 (UP)
target = 0 → next bar return ≤ 0 (DOWN)


The prediction horizon is next-bar direction, suitable for intraday strategies.

🤖 Models Used
1. Logistic Regression

Baseline interpretable model
Fast and stable
Useful for regime impact analysis

2. XGBoost Classifier

Non-linear model
Captures complex feature interactions
Generally higher predictive performance

Both models are:

Trained using time-based splits
Saved using joblib
Loaded dynamically in the Streamlit app

🖥️ Streamlit Application

The Streamlit app (app.py) provides:
CSV upload for OHLCV data
Automatic feature generation
Model selection (Logistic / XGBoost)
Latest market prediction with confidence
Interactive price chart visualization

Run the app locally
streamlit run app.py

🧩 Tech Stack

Python 3.10+
Pandas, NumPy
Scikit-learn
XGBoost
Streamlit
Matplotlib
Joblib

Git & GitHub

👤 Author
Prat
Quantitative Trading & Machine Learning
