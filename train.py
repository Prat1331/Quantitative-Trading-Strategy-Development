import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from features import build_features

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "data/spot_cleaned.csv"

LOG_MODEL_PATH = "models/logistic_model.pkl"
XGB_MODEL_PATH = "models/xgb_model.pkl"
FEATURES_PATH = "models/final_features.pkl"

TRAIN_RATIO = 0.7
RANDOM_STATE = 42

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
df = build_features(df)

# ----------------------------
# TARGET
# ----------------------------
df["target"] = (df["log_return"].shift(-1) > 0).astype(int)

df = df.dropna().copy()

# ----------------------------
# FINAL FEATURES (SAVE THESE)
# ----------------------------
final_features = [
    col for col in df.columns
    if col not in ["target", "date"]
]

X = df[final_features]
y = df["target"]

# Save feature list (CRITICAL)
joblib.dump(final_features, FEATURES_PATH)

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

joblib.dump(xgb_model, XGB_MODEL_PATH)

print("\n✅ Training complete. Models & features saved.")
