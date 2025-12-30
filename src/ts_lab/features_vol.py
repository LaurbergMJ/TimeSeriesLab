from __future__ import annotations
import numpy as np 
import pandas as pd 
from src.ts_lab.volatility import log_returns, future_realized_vol_target, realized_vol

def build_vol_features(close: pd.Series) -> pd.DataFrame:
    """
    Features to predict future realized volatility
    Must use only info available up to time t 
    """

    r = log_returns(close)
    abs_r = r.abs()

    X = pd.DataFrame(index=close.index)

    # Vol level (past)
    for w in [5, 10, 20, 60]:
        X[f"rv_{w}"] = r.shift(1).rolling(w).std()
        X[f"absr_{w}"] = abs_r.shift(1).rolling(w).mean()

    # Vol-of-vol - stability of vol
    for w in [10, 20, 60]:
        X[f"vov_{w}"] = (r.shift(1).rolling(5).std()).rolling(w).std()

    # Trend & drawdown - often related to vol regimes
    for w in [20, 60, 120]:
        ma = close.rolling(w).mean()
        X[f"trend_{w}"] = close / ma - 1.0

    roll_max_252 = close.rolling(252).max()
    X["dd_252"] = close / roll_max_252 - 1.0 

    return X

def make_supervised_vol(
    close: pd.Series,
    target_window: int = 5,
    annualize_target: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    
    X_t: features using data up to t
    y_t: realized vol over t+1... t+target_window
    """

    X = build_vol_features(close)

    #Added in Phase 6.5.1
    X["rv_last"] = realized_vol(close, window=target_window, annualize=annualize_target).shift(1)
    y = future_realized_vol_target(close, window=target_window, annualize=annualize_target)

    df = pd.concat([X, y], axis=1).dropna()
    return df.drop(columns=[y.name]), df[y.name]



