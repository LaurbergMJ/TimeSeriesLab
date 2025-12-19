from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from src.ts_lab.data_io import load_close_data
from src.ts_lab.features import build_features 
from src.ts_lab.split import train_test_split_time
from src.ts_lab.models.linear_regression import make_linear_regression_pipeline
from src.ts_lab.evaluation import regression_report
from src.ts_lab.plotting import plot_lin_reg, plot_folds
from src.ts_lab.walkforward import walk_forward_cv

def main() -> None:
    
    filename = "data/eod_SX5E_index.csv"
    close = load_close_data(filename)

    TEST_SIZE = 0.2
    RUN_WALK_FORWARD = True 
    N_SPLITS = 6

    X, y = build_features(close)

    # simple chronological split 
    X_train, X_test, y_train, y_test = train_test_split_time(
        X, 
        y, 
        test_size=TEST_SIZE)
    

    model = make_linear_regression_pipeline()

    metrics_df, fold_results = walk_forward_cv(model, X, y, n_splits=N_SPLITS)

    print("\n=== Walk-forward results (Linear Regression) ===")
    print(metrics_df)

    print("\nSummary (mean):")
    print(metrics_df[["mae", "rmse", "r2", "directional_accuracy", "corr"]].mean())

    print("\nSummary (std):")
    print(metrics_df[["mae", "rmse", "r2", "directional_accuracy", "corr"]].std())

    plot_folds(
        fold_results,
        title="Linear Regression walk-forward",
        max_cols=2,
        show_scatter=False
    )

    
    # --- snippet used for standard linear regression ---
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # report = regression_report(y_test, y_pred)

    # print("\n=== Linear Regression Baseline ===")
    # for k, v in report.items():
    #     print(f"{k:>22s}: {v:.6f}")
    # plot_lin_reg(y_test, y_pred, label="SX5E index")
    # ---------------------------------------------------

   

if __name__ == "__main__":
    main()
