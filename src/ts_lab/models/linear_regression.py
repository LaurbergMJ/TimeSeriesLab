from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_linear_regression_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())])

