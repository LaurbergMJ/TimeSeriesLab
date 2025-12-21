from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from src.ts_lab.settings import SETTINGS
from src.ts_lab.data_io import load_close_data, list_csv_files
from src.ts_lab.features import make_supervised 
from src.ts_lab.models.linear_regression import make_linear_regression_pipeline
from src.ts_lab.evaluation import regression_report
from src.ts_lab.walkforward import walk_forward_cv_with_baselines
from src.ts_lab.plotting import plot_lin_reg, plot_folds, plot_folds_multi
from src.ts_lab.experiments import evaluate_feature_sets
from src.ts_lab.feature_checks import print_feature_sanity, plot_feature_corr_with_target
from src.ts_lab.split import train_test_split_time

#------------
# Lab Settings
#------------

filename = "data/eod_SX5E_index.csv"
USE_FIRST_CSV_IF_NOT_FOUND = True

TEST_SIZE = 0.2
N_SPLITS = 12
HORIZON = 1
ROLLING_MEAN_WINDOW = 20

# Phase toggles
RUN_SINGLE_SPLIT = False 
RUN_WALK_FORWARD = True 
RUN_PHASE2_FEATURE_COMPARE = True

FEATURE_SETS = [
    "basic",
    "v1_small",        
    "v1_full", 
    "v1_no_trend", 
    "v1_no_vol", 
    "v1_no_dd",
]

def main() -> None:

    close = load_close_data(filename)     
    model = make_linear_regression_pipeline()

    if RUN_SINGLE_SPLIT:
        X, y = make_supervised(close, feature_set="basic", horizon=HORIZON)

        cut = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y.iloc[:cut], y.iloc[cut:]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = regression_report(y_test, y_pred)
        print("\n=== Single split (sanity) ===")
        for key, value in report.items():
            print(f"{key:>22s}: {value:-6f}")

        plt.figure()
        plt.plot(y_test.index, y_test.values, label="actual")
        plt.plot(y_test.index, y_pred, label="pred")
        plt.legend()
        plt.title("Single split sanity check")
        plt.tight_layout()
        plt.show()

    
    if RUN_WALK_FORWARD:
        X, y = make_supervised(close, feature_set="basic", horizon=HORIZON)

        print_feature_sanity(X)
        plot_feature_corr_with_target(X, y, top_n=20)

        metrics_df, fold_results = walk_forward_cv_with_baselines(
            model,
            X,y,
            n_splits=N_SPLITS,
            rolling_mean_window=ROLLING_MEAN_WINDOW,
        )

        print("\n=== Phase 1: Walk-forward results (all models) ===")
        print(metrics_df)

        cols = ["mae", "rmse", "r2", "directional_accuracy", "corr"]
        print("\n=== Phase 1 summary (mean by model) ===")
        print(metrics_df.groupby("model")[cols].std())

        plot_folds_multi(
            fold_results,
            title="Phase 1: Linear Regression vs baselines",
            max_cols=2,
            include_models=[
                "linear_regression",
                "zero",
                "last",
                f"mean_{ROLLING_MEAN_WINDOW}",
            ],
        )

    if RUN_PHASE2_FEATURE_COMPARE:
        summary, per_set = evaluate_feature_sets(
            close=close, 
            model=model,
            feature_sets=FEATURE_SETS,
            horizon=HORIZON,
            n_splits=N_SPLITS,
            rolling_mean_window=ROLLING_MEAN_WINDOW,
        )

        print("\n=== Phase 2 step 2: Feature set comparison (mean across folds) ===")
        print(summary.sort_values(["model", "rmse"]))

        print("\n=== Linear Regression only (best -> worst by RMSE) ===")
        print(
            summary[summary["model"] == "linear_regression"]
            .sort_values("rmse")
        )
    
if __name__ == "__main__":
    main()
