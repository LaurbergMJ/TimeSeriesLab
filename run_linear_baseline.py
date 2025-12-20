from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from src.ts_lab.data_io import load_close_data
from src.ts_lab.features import make_supervised 
from src.ts_lab.split import train_test_split_time
from src.ts_lab.models.linear_regression import make_linear_regression_pipeline
from src.ts_lab.evaluation import regression_report
from src.ts_lab.plotting import plot_lin_reg, plot_folds, plot_folds_multi
from src.ts_lab.walkforward import walk_forward_cv_with_baselines
from src.ts_lab.feature_checks import print_feature_sanity, plot_feature_corr_with_target

def main() -> None:
    
    filename = "data/eod_SX5E_index.csv"
    close = load_close_data(filename)

    TEST_SIZE = 0.2
    RUN_WALK_FORWARD = True 
    N_SPLITS = 6
    ROLLING_MEAN_WINDOW = 20
    FEATURE_SET = "v1" # Choose between "v1" or "basic"
    HORIZON = 1

    X, y = make_supervised(close, feature_set=FEATURE_SET, horizon=HORIZON)
    
    # simple chronological split 
    X_train, X_test, y_train, y_test = train_test_split_time(
        X, 
        y, 
        test_size=TEST_SIZE)
    

    model = make_linear_regression_pipeline()

    metrics_df, fold_results = walk_forward_cv_with_baselines(
        model, 
        X, y, 
        n_splits=N_SPLITS,
        rolling_mean_window=ROLLING_MEAN_WINDOW
        )
        
    #print("\n=== Walk-forward results (All models/baselines) ===")
    #print(metrics_df)

    cols = ["mae", "rmse", "r2", "directional_accuracy", "corr"]

    #print("\nSummary (mean by model) ===")
    #print(metrics_df.groupby("model")[cols].mean())

    #print("\n=== Summary (std by model) === ")
    #print(metrics_df.groupby("model")[cols].std())

    # plot_folds_multi(
    #     fold_results,
    #     title="LR vs baselines (walk-forward)",
    #     max_cols=2,
    #     include_models=["linear_regression", f"mean_{ROLLING_MEAN_WINDOW}"]
    #     #include_models=["linear_regression", "zero", "last", f"mean_{ROLLING_MEAN_WINDOW}"]
    # )

    print_feature_sanity(X)
    plot_feature_corr_with_target(X, y, top_n=20)
    
if __name__ == "__main__":
    main()
