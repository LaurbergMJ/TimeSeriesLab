from __future__ import annotations

import pandas as pd 
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def tune_model_ts(
        model, 
        param_grid: dict,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 6,
        scoring: str = "neg_root_mean_squared_error",
        refit: bool = True,
    ) -> GridSearchCV:
    """
    Time-series safe hyperparameter tuning

    scoring:
     - "neg_root_mean_squared_error" (good default)
     - "neg_mean_absolute_error"
     - "r2"
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring=scoring,
        refit=refit,
        n_jobs=-1,
    )
    gs.fit(X, y)
    return gs