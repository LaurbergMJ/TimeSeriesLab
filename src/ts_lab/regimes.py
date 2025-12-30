from __future__ import annotations

import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def log_returns(close: pd.Series) -> pd.Series:
    r = np.log(close).diff()
    r.name = "log_return"
    return r 

def build_regime_features(close:pd.Series) -> pd.DataFrame:
    """
    Regime features computed from information up to time t
    These are not prediction features, they are for regime labeling/analysis
    """

    r = log_returns(close)
    idx = close.index

    df = pd.DataFrame(index=idx)

    # Volatility level proxies
    df["vol_20"] = r.shift(1).rolling(20).std()
    df["vol_60"] = r.shift(1).rolling(60).std()

    # Trend strength - distance from moving average
    ma_60 = close.rolling(60).mean()
    df["trend_60"] = close / ma_60 - 1.0 

    # Drawdown / stress proxy
    roll_max_252 = close.rolling(252).max()
    df["dd_252"] = close / roll_max_252 - 1.0 # negative in drawdowns

    # Short-term momentum 
    df["r_mean_20"] = r.shift(1).rolling(20).mean()

    return df 

def fit_kmeans_regimes(
        regime_X: pd.DataFrame,
        n_regimes: int = 3,
        random_state: int = 42
    ) -> Pipeline:
    """
    Returns a pipeline: StandardScaler -> KMeans
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_regimes, random_state=random_state, n_init="auto")),
    ])

    pipe.fit(regime_X)
    return pipe

def assign_regimes(
        close: pd.Series,
        n_regimes: int = 3, 
        random_state: int = 42, 
) -> tuple[pd.DataFrame, pd.Series, Pipeline]:
    """
    Returns:
        - regime_X: regime features (cleaned)
        - regime_labels: pd.Series aligned to regime_X index
        - fitted pipeline
    """

    regime_X = build_regime_features(close).dropna()
    pipe = fit_kmeans_regimes(regime_X, n_regimes=n_regimes, random_state=random_state)
    labels = pipe.predict(regime_X)
    regime_labels = pd.Series(labels, index=regime_X.index, name="regime")
    return regime_X, regime_labels, pipe

def pick_crisis_regime(regime_X_train: pd.DataFrame, reg_train: pd.Series) -> int:
    df = regime_X_train.join(reg_train.rename("regime")).dropna()

    # crisis = highest vol_20 mean
    return int(df.groupby("regime")["vol_20"].mean().idxmax())

