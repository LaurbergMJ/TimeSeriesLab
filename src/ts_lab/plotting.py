import matplotlib.pyplot as plt
import math
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
