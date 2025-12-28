from __future__ import annotations
import pandas as pd 
import numpy as np 

def regime_filtered_signal(
    y_pred: pd.Series, 
    regimes: pd.Series, 
    active_regime: int, 
    threshold: float = 0.0,
) -> pd.Series:
    
    """
    Generates a trading signal: 
     - +1 if y_pred > threshold AND regime == active_regime
     - -1 if y_pred < -threshold AND regime == active_regime
     - 0 otherwise
    """

    df = pd.concat([y_pred.rename("pred"), regimes.rename("regime")], axis=1)

    signal = pd.Series(0.0, index=df.index)
    active  = df["regime"] == active_regime 

    signal[active & (df["pred"] > threshold)] == 1.0 
    signal[active & (df["pred"] < -threshold)] == -1.0

    signal.name = "signal"
    return signal

def strategy_returns(
    signal: pd.Series, 
    y_true: pd.Series,
    ) -> pd.Series:
    """
    Strategy return = signal_t * realized_return_{t+1}
    Assumes y_true is already the future return aligned with t
    """

    r = signal * y_true 
    r.name = "strategy_return"
    return r 

def performance_summary(r: pd.Series) -> dict[str, float]:
    """
    Basic performance metrics
    """
    r = r.dropna()
    if len(r) == 0:
        return {"n": 0}
    
    ann_factor = 252 

    out = {
        "n_obs": len(r),
        "mean": r.mean(),
        "std": r.std(),
        "sharpe": (r.mean() / r.std()) * np.sqrt(ann_factor) if r.std() > 0 else np.nan, 
        "hit_rate": (r > 0).mean(),
        "max_drawdown": (r.cumsum() - r.cumsum().cummax()).min(),
    }
    return out 

