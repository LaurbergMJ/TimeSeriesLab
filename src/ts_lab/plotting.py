from __future__ import annotations
import matplotlib.pyplot as plt
import math
import pandas as pd 
from src.ts_lab.walkforward import FoldResult 

def plot_lin_reg(test_values, pred_values, label: str | None ):
    plt.figure()
    plt.plot(test_values.index, test_values.values, label=f"{label} actual")
    plt.plot(test_values.index, pred_values, label=f"{label} predicted")
    plt.title(f"Next-day log return prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_folds(
        fold_results: list[FoldResult],
        title: str = "Walk-forward folds: actual vs. predicted",
        max_cols: int = 2, 
        show_scatter: bool = False
    ) -> None:

    """
    Plots each fold in its own panel, using matplotlib default colors
    """

    n = len(fold_results)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    # Time-series panels
    fig = plt.figure()
    for i, fr in enumerate(fold_results, start=1):
        ax = fig.add_subplot(rows, cols, i)
        ax.plot(fr.y_true.index, fr.y_true.values, label="actual")
        ax.plot(fr.y_pred.index, fr.y_pred.values, label="pred")
        ax.set_title(
            f"Fold {fr.fold} | rmse={fr.metrics['rmse']:.4f} "
            f"| corr={fr.metrics['corr']:.3f}"
        )
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()

    if show_scatter:
        fig2 = plt.figure()
        for i, fr in enumerate(fold_results, start=1):
            ax=fig2.add_subplot(rows, cols, i)
            ax.scatter(fr.y_true.values, fr.y_pred.values, s=10)
            ax.set_title(f"Fold {fr.fold}: pred vs actual")
            ax.set_xlabel("actual")
            ax.set_ylabel("pred")
        fig2.suptitle("Walk-forward folds: scatter")
        fig2.tight_layout()
    
    plt.show()

def plot_folds_multi(
        fold_results: list[FoldResult],
        title: str = "Walk-forward folds: actual vs predictions",
        max_cols: int = 2,
        include_models: list[str] | None = None
) -> None:
    
    """
    For each fold, plot actual plus multiple model/baseline predictions
    """

    n = len(fold_results)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    fig = plt.figure()

    for i, fr in enumerate(fold_results, start=1):
        ax = fig.add_subplot(rows, cols, i)
        ax.plot(fr.y_true.index, fr.y_true.values, label="actual")

        names = list(fr.preds.keys())
        if include_models is not None:
            names = [nm for nm in names if nm in include_models]

        for name in names:
            ax.plot(fr.preds[name].index, fr.preds[name].values, label=name)

        if "linear_regression" in fr.metrics:
            rmse = fr.metrics["linear_regression"]["rmse"]
            corr = fr.metrics["linear_regression"]["corr"]
            ax.set_title(f"Fold {fr.fold} | LR rmse={rmse:.4f} | corr={corr:.3f}")
        
        else:
            ax.set_title(f"Fold {fr.fold}")

        ax.tick_params(axis="x", labelrotation=25)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_regimes_over_price(close: pd.Series, regimes: pd.Series, title: str = "Regimes over price") -> None:
    """
    Simple plot: price line + colored markers by regime
    """

    df = pd.concat([close.rename("close"), regimes.rename("regime")], axis=1).dropna()

    plt.figure()
    plt.plot(df.index, df["close"].values, label="close")

    # Overlay regime points
    for reg in sorted(df["regime"].unique()):
        mask = df["regime"] == reg 
        plt.scatter(df.index[mask], df["close"][mask], s=8, label=f"regime {reg}")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def summarize_regimes(regime_X: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """
    Returns mean regime feature values per regime + counts
    """

    df = pd.concat([regime_X, regimes.rename("regime")], axis=1).dropna()
    out = df.groupby("regime").agg(["mean", "std"])
    counts = df["regime"].value_counts().sort_index()
    out[("meta", "count")] = counts 
    return out 

def summarize_regime_features(regime_X: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """
    Returns mean, std and count of each regime feature per regime
    """

    df = pd.concat([regime_X, regimes.rename("regime")], axis=1).dropna()
    summary = df.groupby("regime").agg(["mean", "std"])
    counts = df["regime"].value_counts().sort_index()
    summary[("meta", "count")] = counts 
    return summary

def plot_single_regime(
    close: pd.Series,
    regimes: pd.Series, 
    target_regime: int, 
    title: str | None = None, 
) -> None:
    df = pd.concat([close.rename("close"), regimes.rename("regime")], axis=1).dropna()
    mask = df["regime"] == target_regime

    plt.figure()
    plt.plot(df.index, df["close"], label="close", alpha=0.5)
    plt.scatter(df.index[mask], df["close"][mask], color="red", s=15, label=f"regime {target_regime}")
    plt.legend()
    plt.title(title or f"Regime {target_regime} highlighted")
    plt.tight_layout()
    plt.show()
