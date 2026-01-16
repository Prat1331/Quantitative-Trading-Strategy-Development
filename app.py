import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------
# Load models
# ------------------
log_model = joblib.load("D:\Quantitative Trading Strategy Development\models\logistic_model.pkl")
xgb_model = joblib.load("D:\Quantitative Trading Strategy Development\models\xgb_model.pkl")

st.set_page_config(page_title="Intraday Regime ML Predictor", layout="wide")

st.title("📈 Regime-Aware Intraday Prediction App")

# ------------------
# Upload data
# ------------------
uploaded_file = st.file_uploader("Upload OHLCV CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default dataset")
    df = pd.read_csv("spot_cleaned.csv")

# ------------------
# Preprocessing
# ------------------
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# EMA
df['ema_fast'] = df['close'].ewm(span=12).mean()
df['ema_slow'] = df['close'].ewm(span=26).mean()
df['ema_slope'] = df['ema_fast'] - df['ema_slow']

# Volatility
VOL_WINDOW = 20
df['rolling_vol'] = df['log_return'].rolling(VOL_WINDOW).std()

vol_median = df['rolling_vol'].median()
df['vol_regime'] = np.where(df['rolling_vol'] > vol_median, 'HIGH_VOL', 'LOW_VOL')
df['trend_regime'] = np.where(df['ema_slope'] > 0, 'UPTREND', 'DOWNTREND')
df['market_regime'] = df['trend_regime'] + "_" + df['vol_regime']

df = df.dropna().copy()

# ------------------
# Feature matrix
# ------------------
features = [
    'log_return',
    'ema_fast',
    'ema_slow',
    'ema_slope',
    'rolling_vol'
]

X = df[features]

# ------------------
# Prediction
# ------------------
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "XGBoost"])

if model_choice == "Logistic Regression":
    preds = log_model.predict(X)
else:
    preds = xgb_model.predict(X)

df['prediction'] = preds

# ------------------
# Latest prediction
# ------------------
latest = df.iloc[-1]

st.subheader("📌 Latest Prediction")
st.write("Market Regime:", latest['market_regime'])
st.write("Prediction (1=UP, 0=DOWN):", int(latest['prediction']))

# ------------------
# Plot
# ------------------
st.subheader("📊 Price Chart with Regime")

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df['close'], label='Close')

ax.set_title("Price Chart")
ax.legend()
st.pyplot(fig)
