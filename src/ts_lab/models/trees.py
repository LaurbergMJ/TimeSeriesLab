from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_random_forest(
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        random_state: int = 42, 
    ) -> Pipeline:

    """
    Random Forest with conservative defaults to reduce overfitting on small series
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

def make_hist_gb(
    max_depth: int = 3,
    learning_rate: float = 0.05,
    max_iter: int = 300,
    min_samples_leaf: int = 20,
    random_state: int = 42,
    ) -> Pipeline:

    """
    Histogram-based gradient boosting (sklearn's best tree model)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )),
    ])

