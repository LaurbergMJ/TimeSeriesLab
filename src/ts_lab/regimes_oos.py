from __future__ import annotations
import pandas as pd 
from sklearn.pipeline import Pipeline
from src.ts_lab.regimes import build_regime_features, fit_kmeans_regimes

def split_by_time(index: pd.Index, train_frac: float = 0.7) -> tuple[pd.Index, pd.Index]:
    n = len(index)
    cut = int(n * train_frac)
    return index[:cut], index[cut:]

def fit_regimes_on_train(
        close: pd.Series, 
        n_regimes: int = 4,
        train_frac: float = 0.7,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, Pipeline, pd.Timestamp]:
    
    """
    Fits regimes on early sample, returns regime features and labels for train + test
    """

    regime_X_all = build_regime_features(close).dropna()
    train_idx, test_idx = split_by_time(regime_X_all.index, train_frac=train_frac)

    X_train = regime_X_all.loc[train_idx]
    X_test = regime_X_all.loc[test_idx]

    pipe = fit_kmeans_regimes(X_train, n_regimes=n_regimes, random_state=random_state)

    reg_train = pd.Series(pipe.predict(X_train), index=X_train.index, name="regime")
    reg_test = pd.Series(pipe.predict(X_test), index=X_test.index, name="regime")

    split_date = test_idx[0]
    return X_train, reg_train, X_test, reg_test, pipe, split_date

