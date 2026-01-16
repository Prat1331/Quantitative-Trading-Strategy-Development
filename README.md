Quantitative Trading Strategy Development
Regime-Aware Intraday Market Prediction

1. Project Objective

The objective of this project is to build a regime-aware machine learning system that predicts the next-bar price direction (UP/DOWN) for intraday financial data using technical indicators and market regime classification.

This project demonstrates:

Financial feature engineering

Market regime detection

Supervised ML model training

Model deployment using Streamlit

2. What Has Been Implemented ✅
Data & Features

Cleaned OHLCV market data

Log returns

Exponential Moving Average (EMA)

EMA slope (trend strength)

Rolling volatility

Trend regime (UPTREND / DOWNTREND)

Volatility regime (HIGH / LOW)

Market regime (trend × volatility)

One-hot encoded regime features

Machine Learning Models

Logistic Regression

XGBoost Classifier

Time-based train/test split

Feature consistency using saved feature list

Model persistence using joblib

Streamlit Application

Upload OHLCV CSV file

Automatic feature generation

Model selection (Logistic / XGBoost)

Latest prediction with confidence score

Price chart visualization

3. Project Structure
Quantitative-Trading-Strategy-Development/
│
├── app.py              # Streamlit application
├── train.py            # Model training pipeline
├── features.py         # Feature engineering logic
├── README.md           # Project documentation
├── requirements.txt    # Dependencies
│
├── data/
│   └── spot_cleaned.csv
│
├── models/
│   ├── logistic_model.pkl
│   ├── xgb_model.pkl
│   └── final_features.pkl

4. How to Run the Project
Train Models
python train.py

Run Streamlit App
streamlit run app.py

5. What Is NOT Implemented Yet 🚧

The following were not part of the current scope but can be added later:

LSTM / Deep learning models

Walk-forward or rolling retraining

Transaction costs & slippage

Portfolio construction & position sizing

Live market data integration

Automated trade execution

Risk management (stop-loss / take-profit)

Hyperparameter optimization

Full backtesting engine

6. Technologies Used

Python

Pandas, NumPy

Scikit-learn

XGBoost

Streamlit

Matplotlib

Joblib

Git & GitHub

7. Disclaimer

This project is for educational and research purposes only.
It does not constitute financial advice and should not be used for live trading without proper validation and risk management.
