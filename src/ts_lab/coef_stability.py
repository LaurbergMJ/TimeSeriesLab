from __future__ import annotations

import numpy as np 
import pandas as pd 
from sklearn.base import clone 
from sklearn.model_selection import TimeSeriesSplit

def collect_fold_coefs(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 6,
    ) -> pd.DataFrame:

    """
    Fits the model per fold and returns a DataFrame: 
    - rows = folds
    - cols = feature names
    Assumes model is a Pipeline with a final step named "model" having coef_
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        m = clone(model)
        m.fit(X_train, y_train)

        estimator = m.named_steps["model"]
        if not hasattr(estimator, "coef_"):
            raise ValueError("Final estimator has no coef_ attribute")
        
        coef = np.asarray(estimator.coef_).ravel()
        rows.append(pd.Series(coef, index=X.columns, name=f"fold_{fold}"))

    return pd.DataFrame(rows)

def coef_stability_summary(coefs: pd.DataFrame) -> pd.DataFrame:
    """
    Stability stats per feature:
     - mean coefficient
     - std coefficient
     - sign flip rate across folds
    """

    mean = coefs.mean(axis=0)
    std = coefs.std(axis=0)

    signs = np.sign(coefs)

    # sign flips: fraction of folds not equal to the majority sign
    majority = signs.mode(axis=0).iloc[0].replace(0.0, np.nan)
    flip_rate = (signs.ne(majority)).mean(axis=0)

    out = pd.DataFrame({
        "coef_mean": mean,
        "coef_std": std, 
        "sign_flip_rate": flip_rate,
    }).sort_values("coef_std", ascending=False)
    return out
    
