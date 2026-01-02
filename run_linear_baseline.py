from __future__ import annotations

from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.ts_lab.settings import SETTINGS
from src.ts_lab.data_io import load_close_data, list_csv_files
from src.ts_lab.features import make_supervised
from src.ts_lab.features_vol import make_supervised_vol 
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
from src.ts_lab.regimes import assign_regimes, pick_crisis_regime
from src.ts_lab.regime_eval import metrics_by_regime
from src.ts_lab.strategy import regime_filtered_signal, strategy_returns, performance_summary, regime_filtered_signal_with_persistence, vol_scaled_weights
from src.ts_lab.regimes_oos import fit_regimes_on_train
from src.ts_lab.features_state import build_state_features
from src.ts_lab.analogs import fit_state_scaler, transform_states, fit_knn, query_analogs, compute_risk_flags, classify_risk_state
from src.ts_lab.forward_outcomes import forward_returns, forward_volatility

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
RUN_PHASE5_STRATEGY = False 
ACTIVE_REGIME = 2
MIN_CONSECUTIVE_DAYS = 3
RUN_PHASE5_OOS = False
TRAIN_FRAC = 0.7
RUN_PHASE6_VOL = False
VOL_TARGET_WINDOW = 5
RUN_PHASE7_ANALOGS = True
RUN_PHASE8_STABILITY = True

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

        # vol series for sizing
        vol_for_sizing_full = regime_X["vol_20"]

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

        # vol for preds
        vol_for_preds = vol_for_sizing_full.reindex(y_pred_all.index)

        regimes_for_preds = regime_labels.reindex(y_pred_all.index)

        by_regime = metrics_by_regime(y_true_all, y_pred_all, regimes_for_preds)
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
        
        # signal = regime_filtered_signal(
        #     y_pred = y_pred_all,
        #     regimes=regimes_for_preds,
        #     active_regime=ACTIVE_REGIME,
        #     threshold=0.0,
        # )

        signal = regime_filtered_signal_with_persistence(
            y_pred=y_pred_all,
            regimes=regimes_for_preds,
            active_regime=ACTIVE_REGIME,
            min_consecutive_days=MIN_CONSECUTIVE_DAYS,
            threshold=0.0,
        )

        weights = vol_scaled_weights(
            signal=signal,
            vol=vol_for_preds,
            target_vol=0.01,
            max_leverage=2.0
        )

        # strategy returns with sizing
        strat_r = weights * y_true_all
        strat_r.name = "strategy return"

        #strat_r = strategy_returns(signal, y_true_all)
        perf = performance_summary(strat_r)

        #active = signal != 0

        active = weights != 0

        strat_r_active = strat_r[active]
        
        print("\nExposure:", float(active.mean()))
        print("\nActive days:", int(active.sum()), "out of", len(signal))

        perf_active = performance_summary(strat_r_active)
        print("\n=== Regime-filtered strategy performance (ACTIVE days only) ===")
        for k,v in perf_active.items():
            print(f"{k:>15s}: {v:.6f}")

        
        perf_all = performance_summary(strat_r)
        print("\n=== Regime-filtered strategy performance (ALL days) ===")
        for k, v in perf_all.items():
            print(f"{k:>15s}: {v:.6f}")


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


    if RUN_PHASE5_OOS:

        # 1) Fit regimes on TRAIN only and predict regimes on TEST
        Xr_train, reg_train, Xr_test, reg_test, pipe, split_date = fit_regimes_on_train(
            close=close,
            n_regimes=N_REGIMES,
            train_frac=TRAIN_FRAC,
            random_state=42,
        )
        crisis_regime = pick_crisis_regime(Xr_train, reg_train)
        print("\n=== Phase 5 step 12: OOS regime test ===")
        print("Split date:", split_date)
        print("Crisis regime (from TRAIN vol_20):", crisis_regime)
        print("TEST regime counts:\n", reg_test.value_counts().sort_index())

        # 2) Build supervised dataset and keep only TEST period (by date)
        X, y = make_supervised(close, feature_set=PHASE5_MODEL_FEATURE_SET, horizon=PHASE5_HORIZON)
        X_test = X.loc[X.index >= split_date]
        y_test = y.loc[y.index >= split_date]

        # 3) Predict with a walk-forward model ONLY on TEST period (rolling CV inside test)
        # this avoids training on future relative to test window
        model = make_ridge(alpha=1.0)

        metrics_df, fold_results = walk_forward_cv_with_baselines(
            model, 
            X_test, y_test,
            n_splits=max(3, min(6, N_SPLITS)),
            rolling_mean_window=ROLLING_MEAN_WINDOW,
            model_name="ridge_oos",
        )

        cols = ["mae", "rmse", "r2", "directional_accuracy", "corr"]
        print("\n=== Phase 5 step 12: OOS walk-forward mean metrics (TEST only) ===")
        print(metrics_df.groupby("model")[cols].mean())

        # Stitch OOS predictions from folds
        all_true, all_pred = [], []
        for fr in fold_results:
            if "ridge_oos" in fr.preds:
                all_true.append(fr.y_true)
                all_pred.append(fr.preds["ridge_oos"])
        y_true_oos = pd.concat(all_true).sort_index()
        y_pred_oos = pd.concat(all_pred).sort_index()

        # 4) Align TEST regimes + vol series to prediction dates
        regimes_oos = reg_test.reindex(y_pred_oos.index).ffill()
        vol_oos = Xr_test["vol_20"].reindex(y_pred_oos.index).ffill()

        # 5) Build signal: crisis-only + persistence + vol scaling
        signal_oos = regime_filtered_signal_with_persistence(
            y_pred=y_pred_oos,
            regimes=regimes_oos,
            active_regime=crisis_regime,
            min_consecutive_days=MIN_CONSECUTIVE_DAYS,
            threshold=0.0
        )

        weight_oos = vol_scaled_weights(
            signal=signal_oos,
            vol=vol_oos,
            target_vol=0.01,
            max_leverage=2.0,
        )

        strat_r_oos = (weight_oos * y_true_oos).rename("strategy_return")

        active = weight_oos != 0
        print("\nExposure (OOS):", float(active.mean()))
        print("Active days (OOS):", int(active.sum()), "out of", len(active))

        perf_active = performance_summary(strat_r_oos[active])
        print("\n=== Phase 5 Step 12: OOS Performance (ACTIVE days only) ===")
        for k,v in perf_active.items():
            print(f"{k:>15s}: {v:.6f}")

        perf_all = performance_summary(strat_r_oos)
        print("\n=== Phase 5 step 12: OOS performance (ALL days) ====")
        for k, v in perf_all.items():
            print(f"{k:>15s}: {v:.6f}")

            
    if RUN_PHASE6_VOL:
        Xv, yv = make_supervised_vol(close, target_window=VOL_TARGET_WINDOW, annualize_target=False)

        print("\n[Sanity]")
        print("corr(rv_last, y):", float(Xv["rv_last"].corr(yv)))

        y_lag = yv.shift(1).reindex(yv.index)
        print("corr(y_lag1, y):", float(y_lag.corr(yv)))


        model = make_ridge(alpha=1.0)

        metrics_df, fold_results = walk_forward_cv_with_baselines(
            model, 
            Xv, yv,
            n_splits=N_SPLITS,
            rolling_mean_window=ROLLING_MEAN_WINDOW,
            model_name="ridge_vol"
        )

        cols = ["mae", "rmse", "r2", "directional_accuracy", "corr"]
        print("\n=== Phase 6: Vol forecasting mean metrics ===")
        print(metrics_df.groupby("model")[cols].mean())

    if RUN_PHASE7_ANALOGS: # Similarity search/analog analysis
        X_state = build_state_features(close)
        print("\n=== Phase 7.1: State features ===")
        print(X_state.describe())

        scaler = fit_state_scaler(X_state)
        Z = transform_states(X_state, scaler)

        nn = fit_knn(Z, n_neighbors=50)

        test_date = Z.index[-1]
        neighbors = query_analogs(
            t=test_date,
            Z=Z,
            nn=nn, 
            min_lookback_days=252,
            k=10
        )

        print("\n=== Phase 7.2: Analog dates ===")
        print("Query date:", test_date)
        print("Analog dates:", neighbors.tolist())

        horizon = 5 

        analog_rets = forward_returns(
            close, 
            start_dates=neighbors,
            horizon=horizon
        )

        analog_vols = forward_volatility(
            close, 
            start_dates=neighbors,
            horizon=horizon
        )

        print("\n=== Phase 7.3: Analog forward outcomes ===")
        print("Forward returns:")
        print(analog_rets)

        print("\nForward volatilities:")
        print(analog_vols)

        print("nSummary (analogs only):")
        print(pd.concat([analog_rets, analog_vols], axis=1).describe())

        # Unconditional comparison
        all_dates = Z.index[:-horizon]

        uncond_rets = forward_returns(
            close, 
            start_dates=all_dates,
            horizon=horizon,
        )

        uncond_vols = forward_volatility(
            close, 
            start_dates=all_dates,
            horizon=horizon
        )

        print("\n=== Phase 7.3: Unconditional outcomes ===")
        print(pd.concat([uncond_rets, uncond_vols], axis=1).describe())


        plt.figure(figsize=(12,5))
        plt.plot(close, label="price", alpha=0.7)

        plt.scatter(
            neighbors,
            close.loc[neighbors],
            color="red",
            label="Analogs",
            zorder=3
        )

        plt.scatter(
            test_date,
            close.loc[test_date],
            color="black",
            marker="x",
            s=100,
            label="Query date",
            zorder=4
        )

        plt.title("Phase 7.4: Analog dates on price chart")
        plt.legend()
        plt.show()

        analog_states = X_state.loc[neighbors]
        query_state = X_state.loc[[test_date]]

        print("\n==== Phase 7.4: Feature comparison ===")
        print(pd.concat([
            query_state.assign(type="query"),
            analog_states.assign(type="analog")
        ]).groupby("type").mean())

        plt.figure(figsize=(10, 4))
        plt.hist(uncond_rets, bins=50, alpha=0.5, label="Unconditional", density=True)
        plt.hist(analog_rets, bins=10, alpha=0.8, label="Analogs", density=True)
        plt.axvline(analog_rets.mean(), color="red", linestyle="--", label="Analog mean")
        plt.axvline(uncond_rets.mean(), color="black", linestyle="--", label="Uncond mean")
        plt.legend()
        plt.title("Phase 7.4: Forward return distribution")
        plt.show()

        # Phase 7.5 
        regime_X, regime_labels, _ = assign_regimes(close, n_regimes=N_REGIMES)

        analog_regimes = regime_labels.reindex(neighbors)

        print("\n === Phase 7.5: Analog regimes ===")
        print(analog_regimes)

        # Phase 7.5.3
        analog_counts = analog_regimes.value_counts().sort_index()
        analog_frac = analog_counts / analog_counts.sum()

        # Unconditional regime distribution - same sample
        uncond_counts = regime_labels.loc[Z.index].value_counts().sort_index()
        uncond_frac = uncond_counts / uncond_counts.sum()

        regime_compare = pd.DataFrame({
            "analog_frac": analog_frac,
            "unconditional_frac": uncond_frac,
        }).fillna(0.0)

        print("\n=== Phase 7.5: Regime distribution (analogs vs unconditional) ===")
        print(regime_compare)

        # Plot
        regime_compare.plot(
            kind="bar",
            figsize=(8, 4),
            title="Phase 7.5: Regime distribution - Analogs vs Unconditional",
        )
        plt.ylabel("Fraction")
        plt.show()

        # Phase 7.6 - Time-varying analogs --- 
        query_dates = Z.index[::20]
        query_dates = query_dates[query_dates >= Z.index[0] + pd.Timedelta(days=500)]

        records = []
        horizon = 5
        k = 10

        for t in query_dates:
            analogs_t = query_analogs(
                t=t,
                Z=Z,
                nn=nn,
                min_lookback_days=252,
                k=k
            )

            if len(analogs_t) < k:
                continue 

            # Forward outcomes
            fwd_ret = forward_returns(close, analogs_t, horizon=horizon)
            fwd_vol = forward_volatility(close, analogs_t, horizon=horizon)

            # Regime composition
            reg_t = regime_labels.reindex(analogs_t).dropna()
            dominant_regime = Counter(reg_t).most_common(1)[0][0]

            records.append({
                "date": t,
                "analog_ret_mean": fwd_ret.mean(),
                "analog_ret_median": fwd_ret.median(),
                "analog_vol_mean": fwd_vol.mean(),
                "dominant_regime": dominant_regime
            })

        analog_ts = pd.DataFrame(records).set_index("date")
        print("\n=== Phase 7.6: Rolling analog summary ===")
        print(analog_ts.head())

        # 7.6.3 - Visualization
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(analog_ts.index, analog_ts["analog_ret_mean"], label="Analog mean fwd return")
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)

        ax.set_title("Phase 7.6: Time-varying analog forward return expectations")
        ax.legend()
        plt.show()

        # Rolling analog volatility
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(analog_ts.index, analog_ts["analog_vol_mean"], label="Analog mean fwd vol")
        ax.set_title("Phase 7.6: Time-varying analog volatility expectation")
        ax.legend()
        plt.show()


        # Dominant regime over time
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(analog_ts.index, analog_ts["dominant_regime"], drawstyle="steps-post")
        ax.set_title("Phase 7.6: Dominant analog regime over time")
        ax.set_ylabel("Regime")
        plt.show()

        # Classify Risk States
        risk_flags = compute_risk_flags(
            analog_ts, 
            uncond_ret_mean=uncond_rets.mean(),
            uncond_vol_median=uncond_vols.median(),
        )

        risk_flags["risk_state"] = risk_flags.apply(classify_risk_state, axis=1)

        print("\n=== Phase 7.7: Risk state snapshot ===")
        print(risk_flags.tail())

    if RUN_PHASE8_STABILITY:

        time_splits = {
            "early": ("2010-01-01", "2015-12-31"),
            "mid": ("2016-01-01", "2020-12-31"),
            "late": ("2021-01-01", "2025-12-31")
        }

        # Step 8.1.2 - Regime stability
        print("\n === Phase 8.1: Regime stability by subsample ===")

        for name, (start, end) in time_splits.items():
            mask = (regime_labels.index >= start) & (regime_labels.index <= end)
            frac = regime_labels.loc[mask].value_counts(normalize=True).sort_index()

            print(f"\n{name.upper()} sample:")
            print(frac)

        # Step 8.1.3 - Analog outcome stability across time
        print("\n=== Phase 8.1: Analog outcome stability ===")

        for name, (start, end) in time_splits.items():
            dates = analog_ts.index 
            mask = (dates >= start) & (dates <= end)
            sub = analog_ts.loc[mask]

            if  len(sub) <5:
                print(f"\n{name.upper()}: insufficient data")
                continue 

            print(f"\n{name.upper()} sample:")
            print(sub[["analog_ret_mean", "analog_vol_mean"]].describe())

        # Step 8.1.4 - Risk-state stability 
        print("\n=== Phase 8.1: Risk-state stability ===")

        for name, (start, end) in time_splits.items():
            mask = (risk_flags.index >= start) & (risk_flags.index <= end)
            freq = risk_flags.loc[mask, "risk_state"].value_counts(normalize=True)

            print(f"\n{name.upper()} sample:")
            print(freq)

        # Step 8.4.1 - Implement random analog comparison
        print("\n=== Phase 8.4.2: Null test - random analogs ===")

        rng = np.random.default_rng(42)

        horizon = 5
        k = 10
        n_trials = 100 

        null_ret_means = []
        null_vol_means = []

        valid_dates = Z.index[:-252]

        for _ in range(n_trials):
            random_dates = rng.choice(valid_dates, size=k, replace=False)

            ret = forward_returns(close, random_dates, horizon=horizon)
            vol = forward_volatility(close, random_dates, horizon=horizon)

            null_ret_means.append(ret.mean())
            null_vol_means.append(vol.mean())

        null_ret_means = np.array(null_ret_means)
        null_vol_means = np.array(null_vol_means)

        print("Null analog return mean:")
        print(pd.Series(null_ret_means).describe())

        print("\nNull analog vol mean:")
        print(pd.Series(null_vol_means).describe())
        
        # Step 8.4.3 - Shuffle feature rows
        print("\n=== Phase 8.4: Null test - shuffled state space ===")

        Z_shuffled = Z.sample(frac=1.0, random_state=42)

        nn_null = fit_knn(Z_shuffled, n_neighbors=50)

        null2_ret_means = []

        for t in query_dates[-20:]:
            try:
                analogs_null = query_analogs(
                    t=t,
                    Z=Z_shuffled,
                    nn=nn_null,
                    min_lookback_days=252,
                    k=k,
                )
                if len(analogs_null) < k:
                    continue

                ret = forward_returns(close, analogs_null, horizon=horizon)
                null2_ret_means.append(ret.mean())
            except Exception:
                continue

        print("Shuffled-state analog return means:")
        print(pd.Series(null2_ret_means).describe())


        














        




        




if __name__ == "__main__":
    main()
