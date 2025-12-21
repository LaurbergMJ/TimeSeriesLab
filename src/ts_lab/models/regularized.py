from __future__ import annotations

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_ridge(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha))
    ])

def make_lasso(alpha: float = 0.001, max_iter: int = 50_000) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=alpha, max_iter=max_iter))
    ])

def make_elasticnet(alpha: float = 0.001, l1_ratio: float = 0.5, max_iter: int = 50_000) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter))
    ])

