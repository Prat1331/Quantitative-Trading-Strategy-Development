import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from features import build_features

# ------------------
# Load models & metadata
# ------------------
log_model = joblib.load("models/logistic_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
final_features = joblib.load("models/final_features.pkl")

st.set_page_config(
    page_title="Intraday Regime ML Predictor",
    layout="wide"
)

st.title("📈 Regime-Aware Intraday Prediction App")

# ------------------
# Upload data
# ------------------
uploaded_file = st.file_uploader("Upload OHLCV CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default dataset")
    df = pd.read_csv("data/spot_cleaned.csv")

# ------------------
# Date handling
# ------------------
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

# ------------------
# Feature engineering (IDENTICAL to training)
# ------------------
df = build_features(df)

# Align features EXACTLY as during training
X = df.reindex(columns=final_features, fill_value=0)

# ------------------
# Model selection
# ------------------
model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "XGBoost"]
)

if model_choice == "Logistic Regression":
    preds = log_model.predict(X)
    probs = log_model.predict_proba(X)[:, 1]
else:
    preds = xgb_model.predict(X)
    probs = xgb_model.predict_proba(X)[:, 1]

df["prediction"] = preds
df["confidence"] = probs

# ------------------
# Latest prediction
# ------------------
latest = df.iloc[-1]

st.subheader("📌 Latest Prediction")
st.write("Market Regime:", latest.get("market_regime", "N/A"))
st.write("Prediction:", "⬆️ UP" if latest["prediction"] == 1 else "⬇️ DOWN")
st.write("Confidence:", f"{latest['confidence']:.2%}")

# ------------------
# Plot price
# ------------------
st.subheader("📊 Price Chart")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["close"], label="Close Price")

ax.set_title("Price Chart")
ax.legend()
st.pyplot(fig)
