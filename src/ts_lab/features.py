from __future__ import annotations
import pandas as pd
import numpy as np

def compute_returns(close: pd.Series) -> pd.Series:
    r = np.log(close).diff()
    r.name = "log_returns"
    return r

def _zscore(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window).mean()
    sig = x.rolling(window).std()
    return (x - mu) / sig 

def build_features_basic(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:

    r = compute_returns(close)

    X = pd.DataFrame(index=close.index)
    X["r_1"] = r.shift(0)
    X["r_2"] = r.shift(1)
    X["r_3"] = r.shift(2)

    X["vol_5"] = r.rolling(5).std()
    X["vol_20"] = r.rolling(20).std()

    ma_20 = close.rolling(20).mean()
    X["trend_20"] = (close / ma_20) - 1.0

    return X

def build_features_v1(close: pd.Series) -> pd.DataFrame:
    """
    - feature pack using only info available up to time t
    - target will be t+1, so features are safe
    """

    r = compute_returns(close)
    abs_r = r.abs()

    X = pd.DataFrame(index=close.index)

    # --- Lagged returns (momentum'ish)
    for lag in [0, 1, 3, 4, 9, 19]: # Corresponds to 1, 2, 3, 5, 10, 20 day lags!
        X[f"r_lag_{lag+1}"] = r.shift(lag)

    # --- Rolling mean returns (slow drift)
    for w in [5, 20, 60]:
        X[f"r_mean_{w}"] = r.shift(1).rolling(w).mean() # shift(1) ensures strictly past


    # --- Rolling volatility proxies
    for w in [5, 20, 60]:
        X[f"r_vol_{w}"] = abs_r.shift(1).rolling(w).std() 
        X[f"absr_mean_{w}"] = abs_r.shift(1).rolling(w).mean()

    # --- Trend features (distance to moving average)
    for w in [20, 60, 120]:
        ma = close.shift(0).rolling(w).mean()
        X[f"trend_ma_{w}"] = close / ma - 1.0

    # --- Drawdown-style feature: distance from rolling max
    for w in [60, 252]:
        roll_max = close.shift(0).rolling(w).max()
        X[f"dd_{w}"] = close / roll_max - 1.0 # negative in drawdowns

    # --- Normalized features (z-scores)
    X[f"r_z_20"] = _zscore(r.shift(1), 20)
    X["trend_z_60"] = _zscore((close / close.rolling(60).mean() - 1.0), 60)

    return X

def make_supervised(
        close: pd.Series,
        feature_set: str = "basic",
        horizon: int = 1
    ) -> tuple[pd.DataFrame, pd.Series]:

    """
    Predict future log return at t+horizon from features at time t
    """

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    
    if feature_set == "basic":
        X = build_features_basic(close)
    elif feature_set == "v1":
        X = build_features_v1(close)
    else:
        raise ValueError("feature_set must be 'basic' or 'v1' ")
    
    y = compute_returns(close).shift(-horizon)
    y.name = f"target_log_return_t+{horizon}"

    df = pd.concat([X, y], axis=1).dropna()
    
    return df.drop(columns=[y.name]), df[y.name]


        


