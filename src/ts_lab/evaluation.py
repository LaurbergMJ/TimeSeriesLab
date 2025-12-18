from __future__ import annotations

import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

def directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true.values) == np.sign(y_pred)))

def regression_report(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": directional_accuracy(y_true, y_pred)
    }

