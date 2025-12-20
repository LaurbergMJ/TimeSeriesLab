from __future__ import annotations
import pandas as pd 
from src.ts_lab.features import make_supervised
from src.ts_lab.walkforward import walk_forward_cv_with_baselines

def evaluate_feature_sets(
        close: pd.Series,
        model, 
        feature_sets: list[str],
        horizon: int, 
        n_splits:  int, 
        rolling_mean_window: int=20
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Returns
    - summary: one row per feature_set per model/baseline, with mean metrics across folds
    - per_set_metrics: dict[feature_set] -> full per_fold metrics_df
    """

    summaries = []
    per_set_metrics: dict[str, pd.DataFrame] = {} 

    for fs in feature_sets:
        X, y = make_supervised(close, feature_set=fs, horizon=horizon)
        metrics_df, _ = walk_forward_cv_with_baselines(
            model, X, y, 
            n_splits = n_splits,
            rolling_mean_window=rolling_mean_window
        )
        per_set_metrics[fs] = metrics_df

        cols = ["mae", "rmse", "r2", "directional_accuracy", "corr"]
        g = metrics_df.groupby("model")[cols].mean().reset_index()
        g.insert(0, "feature_set", fs)
        g.insert(1, "horizon", horizon)
        summaries.append(g)

    summary = pd.concat(summaries, ignore_index=True)
    return summary, per_set_metrics

