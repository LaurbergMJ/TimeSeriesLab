from __future__ import annotations

import numpy as np 
import pandas as pd 

from src.ts_lab.evaluation import regression_report

def metrics_by_regime(
    y_true: pd.Series,
    y_pred: pd.Series,
    regimes: pd.Series
) -> pd.DataFrame:
    
    """
    Compute metrics within each regime
    Assumes all inputs align by index (dates)
    """

    df = pd.concat(
        [y_true.rename("y_true"), y_pred.rename("y_pred"), regimes.rename("regime")], 
        axis=1).dropna()

    rows = []

    for reg in sorted(df["regime"].unique()):
        sub = df[df["regime"] == reg]
        met = regression_report(sub["y_true"], sub["y_pred"].values)
        rows.append({"regime": int(reg), "n": len(sub), **met})
    
    return pd.DataFrame(rows).sort_values("regime") 
    
    
