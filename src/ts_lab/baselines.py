from __future__ import annotations

import numpy as np 
import pandas as pd 
from src.ts_lab.evaluation import regression_report

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

def pred_zero(y_test: pd.Series) -> pd.Series:
    return pd.Series(0.0, index=y_test.index, name="zero")

def pred_last(y_all: pd.Series, test_index: pd.Index) -> pd.Series:
    y_shift = y_all.shift(1)
    pred = y_shift.reindex(test_index)
    pred.name = "last"
       
    return pred

def pred_rolling_mean(y_all: pd.Series, test_index: pd.Index, window: int=20) -> pd.Series:

    """
    Docstring for pred_rolling_mean
    
    Rolling mean forecast using only past date
    y_all should contain train+test y in chronological order
    
    For each test date t: pred(t) = mean(y[t-window ... t-1])
    """

    pred = y_all.shift(1).rolling(window).mean().loc[test_index].fillna(0.0)
    pred.name = f"mean_{window}"
    return pred 


