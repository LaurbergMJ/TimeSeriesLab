from __future__ import annotations
import numpy as np
import pandas as pd  
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass
from src.ts_lab.evaluation import regression_report
from src.ts_lab.baselines import pred_zero, pred_last, pred_rolling_mean


@dataclass
class FoldResult:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    y_true: pd.Series 
    preds: dict[str, pd.Series]
    metrics: dict[str, float] 

def walk_forward_cv_with_baselines(
        model, 
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        rolling_mean_window: int = 20, 
        model_name: str = "model",
    ) -> tuple[pd.DataFrame, list[FoldResult]]:

    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    rows: list[dict] = []
    fold_results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        m = clone(model)
        m.fit(X_train, y_train)
        y_pred_main = pd.Series(m.predict(X_test), index=y_test.index, name=model_name)

        # --- Baselines (built using only y)
        # Compute rolling mean using only past data

        y_all = pd.concat([y_train, y_test])

        preds: dict[str, pd.Series] = {
            model_name: y_pred_main,
            "y_lag1": y_all.shift(1).reindex(y_test.index),
            f"mean_{rolling_mean_window}": pred_rolling_mean(y_all=y_all, test_index=y_test.index, window=rolling_mean_window)         
            
            # "zero": pred_zero(y_test),
            # "last": pred_last(y_all=y_all, test_index=y_test.index),
            # f"mean_{rolling_mean_window}": pred_rolling_mean(
            #     y_all=y_all, test_index=y_test.index, window=rolling_mean_window
            # ),
        }

        if "rv_last" in X_test.columns:
            preds["rv_last"] = X_test["rv_last"]

        metrics: dict[str, dict[str, float]] = {}

        for name, pred in preds.items():
            met = regression_report(y_test, pred.values)
            metrics[name] = met 
            rows.append({
                "fold": fold,
                "model": name, 
                "train_end": X_train.index[-1],
                "test_start": X_test.index[0],
                "test_end": X_test.index[-1],
                **met
            })
                   
        fold_results.append(
            FoldResult(
                fold=fold,
                train_end=X_train.index[-1],
                test_start=X_test.index[0],
                test_end=X_test.index[-1],
                y_true=y_test,
                preds=preds,
                metrics=metrics,
            )
        )

    metrics_df = pd.DataFrame(rows)
    return metrics_df, fold_results