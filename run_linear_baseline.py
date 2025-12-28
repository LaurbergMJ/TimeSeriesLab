from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.ts_lab.settings import SETTINGS
from src.ts_lab.data_io import load_close_data, list_csv_files
from src.ts_lab.features import make_supervised 
from src.ts_lab.models.linear_regression import make_linear_regression_pipeline
from src.ts_lab.models.regularized import make_ridge, make_lasso, make_elasticnet
from src.ts_lab.models.trees import make_random_forest, make_hist_gb
from src.ts_lab.evaluation import regression_report
from src.ts_lab.walkforward import walk_forward_cv_with_baselines
from src.ts_lab.plotting import plot_lin_reg, plot_folds, plot_folds_multi, plot_regimes_over_price, summarize_regimes, summarize_regime_features, plot_single_regime
from src.ts_lab.experiments import evaluate_feature_sets
from src.ts_lab.feature_checks import print_feature_sanity, plot_feature_corr_with_target
from src.ts_lab.split import train_test_split_time
from src.ts_lab.tuning import tune_model_ts
from src.ts_lab.coef_stability import collect_fold_coefs, coef_stability_summary
from src.ts_lab.regimes import assign_regimes
from src.ts_lab.regime_eval import metrics_by_regime
from src.ts_lab.strategy import regime_filtered_signal, strategy_returns, performance_summary

#-------------
# Lab Settings
#-------------

filename = "data/eod_SX5E_index.csv"
USE_FIRST_CSV_IF_NOT_FOUND = True

TEST_SIZE = 0.2
N_SPLITS = 6
HORIZON = 5
ROLLING_MEAN_WINDOW = 20

# Phase toggles
RUN_SINGLE_SPLIT = False 
RUN_WALK_FORWARD = True 
RUN_PHASE2_FEATURE_COMPARE = False
RUN_PHASE3_TUNING = False
RUN_PHASE3_COEF_STABILITY = False 
PHASE3_FEATURE_SET = "basic" 
PHASE3_SCORING = "neg_root_mean_squared_error"
RUN_PHASE4_TREES = False 
PHASE4_FEATURE_SET = "v1_small"
PHASE4_HORIZON = 1
RUN_PHASE5_REGIMES = True 
N_REGIMES = 4
PHASE5_MODEL_FEATURE_SET = "basic"
PHASE5_HORIZON = 1
RUN_PHASE5_STRATEGY = True 
ACTIVE_REGIME = 2

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
            summary[summary["model"] == "linear_regression"].sort_values("rmse")
        )

    if RUN_PHASE3_TUNING or RUN_PHASE3_COEF_STABILITY:

        X, y = make_supervised(close, feature_set=PHASE3_FEATURE_SET, horizon=HORIZON)

        # 1) Tune Ridge model
        ridge = make_ridge(alpha=1.0)
        ridge_grid = {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1_000]}
        gs_ridge = tune_model_ts(ridge, ridge_grid, X, y, n_splits=N_SPLITS, scoring=PHASE3_SCORING)
        print("\n=== Phase 3: Ridge tuning ===")
        print("Best params:", gs_ridge.best_params_)
        print("Best CV score:", gs_ridge.best_score_)

        # 2) Tune Lasso 
        lasso = make_lasso(alpha=0.001)
        lasso_grid = {"model__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
        gs_lasso = tune_model_ts(lasso, lasso_grid, X, y, n_splits=N_SPLITS, scoring=PHASE3_SCORING)
        print("\n=== Phase 3: Lasso tuning ===")
        print("Best params:", gs_lasso.best_params_)
        print("Best CV score:", gs_lasso.best_score_)

        # 3) Tune ElasticNet
        enet = make_elasticnet(alpha=0.001, l1_ratio=0.5)
        enet_grid = {
            "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        gs_enet = tune_model_ts(enet, enet_grid, X, y, n_splits=N_SPLITS, scoring=PHASE3_SCORING)
        print("\n=== Phase 3: ElasticNet tuning ===")
        print("Best params:", gs_enet.best_params_)
        print("Best CV score", gs_enet.best_score_)

        # --- Evaluate the tuned models (walk-forward)
        tuned_models = {
            "ridge_tuned": gs_ridge.best_estimator_,
            "lasso_tuned": gs_lasso.best_estimator_,
            "enet_tuned": gs_enet.best_estimator_,
        }

        for name, m in tuned_models.items():
            metrics_df, _ = walk_forward_cv_with_baselines(
                m, X, y,
                n_splits=N_SPLITS,
                rolling_mean_window=ROLLING_MEAN_WINDOW
            )
            cols = ["mae", "rmse", "r2", "directional_accuracy", "corr"]
            print(f"\n=== {name} walk-forward summary (mean by model) ===")
            print(metrics_df.groupby("model")[cols].mean())

        # --- Coefficient stability
        if RUN_PHASE3_COEF_STABILITY:
            coefs = collect_fold_coefs(gs_ridge.best_estimator_, X, y, n_splits=N_SPLITS)
            summary = coef_stability_summary(coefs)
            print("\n=== Phase 3: Ridge coefficient stability (top 20 by coef_std) ===")
            print(summary.head(20))

    if RUN_PHASE4_TREES:

        X, y = make_supervised(
            close, 
            feature_set=PHASE4_FEATURE_SET,
            horizon=PHASE4_HORIZON,
        )

        tree_models = {
            "random_forest": make_random_forest(),
            "hist_gb": make_hist_gb(),
        }

        for name, model_tree in tree_models.items():
            print(f"\n==== Phase 4: {name} (walk-forward) ===")

            metrics_df, fold_results = walk_forward_cv_with_baselines(
                model_tree,
                X, y,
                n_splits=N_SPLITS,
                rolling_mean_window=ROLLING_MEAN_WINDOW,
            )
        
            cols = ["mae", "rmse", "r2", "directional_accuracy", "corr"]
            print(metrics_df.groupby("model")[cols].mean())

            plot_folds_multi(
                fold_results,
                title=f"Phase 4: {name} vs baselines",
                include_models=[
                    name,
                    "zero",
                    "mean_20",
                ],
            )

    if RUN_PHASE5_REGIMES:
        
        # 1) Fit regimes on the full history (first pass)
        regime_X, regime_labels, _ = assign_regimes(close, n_regimes=N_REGIMES)
        print("\n=== Phase 5: Regime summary (feature means/std + counts) ===")
        print(summarize_regimes(regime_X, regime_labels))

        plot_regimes_over_price(close, regime_labels, title=f"Phase 5: KMeans regimes (K={N_REGIMES})")

        # 2) Build supervised dataset for forecasting task
        X, y = make_supervised(close, feature_set=PHASE5_MODEL_FEATURE_SET, horizon=PHASE5_HORIZON)

        # align regimes to the supervised target index
        regimes_aligned = regime_labels.reindex(X.index)

        # 3) Run walk-forward for the chosen model 
        model = make_ridge(alpha=1.0)
        metrics_df, fold_results = walk_forward_cv_with_baselines(
            model,
            X, y,
            n_splits=N_SPLITS,
            rolling_mean_window=ROLLING_MEAN_WINDOW,
            model_name="ridge"
        )

        # 4) Evaluate per-fold and regime-conditioned 
        print("\n=== Phase 5: Walk-forward mean metrics ===")
        cols = ["mae", "rmse", "r2", "directional_accuracy", "corr"]
        print(metrics_df.groupby("model")[cols].mean())

        # Regime-conditioned metrics using the stitched predictions from folds

        all_true = []
        all_pred = []
        for fr in fold_results:
            if "ridge" in fr.preds:
                all_true.append(fr.y_true)
                all_pred.append(fr.preds["ridge"])

        y_true_all = pd.concat(all_true).sort_index()
        y_pred_all = pd.concat(all_pred).sort_index()

        regimes_for_preds = regime_labels.reindex(y_pred_all.index)

        by_regime = metrics_by_regime(y_true_all, y_pred_all, regimes_aligned)
        print("\n=== Phase 5: Regime-conditioned metrics (ridge) ===")
        print(by_regime)

        
        # Area for regime validation
        summary = summarize_regime_features(regime_X, regime_labels)
        print("\n=== Phase 5 step 8: Regime feature summary")
        print(summary)

        plot_single_regime(close, 
                           regime_labels,
                           target_regime=2,
                           title="Phase 5 step 8: Stress regime (K=4)")

        print("\n=== Phase 5 step 8: Regime counts by year ===")
        print(regime_labels.groupby(regime_labels.index.year).value_counts().unstack(fill_value=0))
    
    if RUN_PHASE5_STRATEGY:
        signal = regime_filtered_signal(
            y_pred = y_pred_all,
            regimes=regimes_for_preds,
            active_regime=ACTIVE_REGIME,
            threshold=0.0,
        )

        strat_r = strategy_returns(signal, y_true_all)
        perf = performance_summary(strat_r)

        print("\n=== Phase 5 step 9: Regime-filtered strategy performance ===")
        for k, v in perf.items():
            print(f"{k:>15s}: {v:.6f}")

        # Compare to always-on strategy
        always_on_signal = pd.Series(
            np.sign(y_pred_all),
            index=y_pred_all.index,
            name="always_on_signal",
        )
        always_on_r = strategy_returns(always_on_signal, y_true_all)
        always_on_perf = performance_summary(always_on_r)

        print("\n=== Always-on Ridge strategy (comparison) ===")
        for k,v in always_on_perf.items():
            print(f"{k:>15s}: {v:.6f}")


if __name__ == "__main__":
    main()
