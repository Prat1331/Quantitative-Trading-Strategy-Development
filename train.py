import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "data/spot_cleaned.csv"
LOG_MODEL_PATH = "models/logistic_model.pkl"
XGB_MODEL_PATH = "models/xgb_model.pkl"

EMA_FAST = 12
EMA_SLOW = 26
VOL_WINDOW = 20
TRAIN_RATIO = 0.7
RANDOM_STATE = 42

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

# Ensure datetime index if present
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
df["log_return"] = np.log(df["close"] / df["close"].shift(1))

df["ema_fast"] = df["close"].ewm(span=EMA_FAST).mean()
df["ema_slow"] = df["close"].ewm(span=EMA_SLOW).mean()
df["ema_slope"] = df["ema_fast"] - df["ema_slow"]

df["rolling_vol"] = df["log_return"].rolling(VOL_WINDOW).std()

# Regime detection
vol_median = df["rolling_vol"].median()
df["vol_regime"] = np.where(df["rolling_vol"] > vol_median, "HIGH_VOL", "LOW_VOL")
df["trend_regime"] = np.where(df["ema_slope"] > 0, "UPTREND", "DOWNTREND")
df["market_regime"] = df["trend_regime"] + "_" + df["vol_regime"]

# ----------------------------
# TARGET
# ----------------------------
df["target"] = (df["log_return"].shift(-1) > 0).astype(int)

df = df.dropna().copy()

# ----------------------------
# FEATURES
# ----------------------------
FEATURES = [
    "log_return",
    "ema_fast",
    "ema_slow",
    "ema_slope",
    "rolling_vol",
]

X = df[FEATURES]
y = df["target"]

# ----------------------------
# TIME-BASED SPLIT
# ----------------------------
split_idx = int(len(df) * TRAIN_RATIO)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# ----------------------------
# LOGISTIC REGRESSION
# ----------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression Results")
print(classification_report(y_test, y_pred_log))

# Save model
joblib.dump(log_model, LOG_MODEL_PATH)

# ----------------------------
# XGBOOST
# ----------------------------
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Results")
print(classification_report(y_test, y_pred_xgb))

# Save model
joblib.dump(xgb_model, XGB_MODEL_PATH)

print("\nTraining complete. Models saved.")
