from __future__ import annotations

import numpy as np 
import pandas as pd 
from ts_lab.evaluation import regression_report

def baseline_zero(y_test: pd.Series) -> dict:
    y_pred = np.zeros(len(y_test))
    return regression_report(y_test, y_pred)

def baseline_last_return(y_test: pd.Series) -> dict:
    y_pred = y_test.shift(1).fillna(0.0).values 
    return regression_report(y_test, y_pred)

def baseline_rolling_mean(y_train: pd.Series, y_test: pd.Series, window: int=20) -> dict:
    history = pd.concat([y_train, y_test])
    pred = history.shift(1).rolling(window).mean().loc[y_test.index].fillna(0.0).values 
    return regression_report(y_test, pred)


