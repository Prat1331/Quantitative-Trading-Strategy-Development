import numpy as np
import pandas as pd

def build_features(df):
    df = df.copy()

    # Log return
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # EMA + slope
    df["ema"] = df["close"].ewm(span=20).mean()
    df["ema_slope"] = df["ema"].diff()

    # Rolling volatility
    df["rolling_vol"] = df["log_return"].rolling(20).std()

    # Regimes
    df["trend_regime"] = np.where(df["ema_slope"] > 0, "UPTREND", "DOWNTREND")
    vol_median = df["rolling_vol"].median()
    df["vol_regime"] = np.where(df["rolling_vol"] > vol_median, "HIGH_VOL", "LOW_VOL")
    df["market_regime"] = df["trend_regime"] + "_" + df["vol_regime"]

    # One-hot encode regimes
    df = pd.get_dummies(
        df,
        columns=["trend_regime", "vol_regime", "market_regime"],
        drop_first=False
    )

    # ⚠️ DO NOT select features here
    return df.dropna()
