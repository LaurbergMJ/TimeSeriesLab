from __future__ import annotations
import pandas as pd

def train_test_split_time(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    
    n = len(X)
    cut = int(n * (1-test_size))

    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    return X_train, X_test, y_train, y_test

