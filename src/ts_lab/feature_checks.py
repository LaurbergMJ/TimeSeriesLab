from __future__ import annotations
import matplotlib.pyplot as plt 
import pandas as pd 

def print_feature_sanity(X: pd.DataFrame, top_corr: int = 10) -> None:
    missing = X.isna().mean().sort_values(ascending=False)
    print("\n=== Feature missingness (top) ===")
    print(missing.head(15))

    # Basic variance sanity
    var = X.var(numeric_only=True).sort_values()
    print("\n=== Lowest-variance features (top) ===")
    print(var.head(10))

    # Correlations among features can explode; show strongest absolute pairs vs target later
    print(f"\n[INFO] X shape: {X.shape}")

def plot_feature_corr_with_target(X: pd.DataFrame, y: pd.Series, top_n: int = 20) -> None:
    df = pd.concat([X, y.rename("target")], axis=1).dropna() 
    corr = df.corr(numeric_only=True)["target"].drop("target").sort_values(key=lambda s: s.abs(), ascending=False)

    top = corr.head(top_n).sort_values()
    plt.figure()
    plt.barh(top.index, top.values)
    plt.title(f"Top {top_n} feature correlations with target")
    plt.tight_layout()
    plt.show()


     