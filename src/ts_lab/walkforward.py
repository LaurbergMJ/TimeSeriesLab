from __future__ import annotations
import numpy as np
import pandas as pd  
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass
from src.ts_lab.evaluation import regression_report

@dataclass
class FoldResult:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: dict[str, float]
    y_true: pd.Series 
    y_pred: pd.Series 

def walk_forward_cv(
        model, 
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> tuple[pd.DataFrame, list[FoldResult]]:

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict] = []
    fold_results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        m = clone(model)
        m.fit(X_train, y_train)
        y_pred_arr = m.predict(X_test)

        y_pred = pd.Series(y_pred_arr, index=y_test.index, name="y_pred")
        metrics = regression_report(y_test, y_pred_arr)

        rows.append({
            "fold": fold, 
            "train_end": X_train.index[-1], 
            "test_start": X_test.index[0],
            "test_end": X_test.index[-1],
            **metrics
        }) 

        fold_results.append(
            FoldResult(
                fold=fold,
                train_end=X_train.index[-1],
                test_start=X_test.index[0],
                test_end=X_test.index[-1],
                metrics=metrics,
                y_true=y_test,
                y_pred=y_pred
            )
        )

    metrics_df = pd.DataFrame(rows)
    return metrics_df, fold_results