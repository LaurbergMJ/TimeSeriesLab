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
    active = df["regime"] == active_regime 

    signal[active & (df["pred"] > threshold)] = 1.0 
    signal[active & (df["pred"] < -threshold)] = -1.0

    signal.name = "signal"
    return signal

def regime_filtered_signal_with_persistence(
    y_pred: pd.Series, 
    regimes: pd.Series,
    active_regime: int, 
    min_consecutive_days: int = 2,
    threshold: float = 0.0, 
    ) -> pd.Series:

    """
    Same as regime_filtered_signal, but only active if regime persists >= N days
    """
    pred = y_pred.astype(float)
    reg = regimes.astype(int)

    persisted = regime_persistence_mask(reg, active_regime, min_consecutive_days=min_consecutive_days)

    s = pd.Series(0.0, index=pred.index, name="signal")
    signed = np.sign(pred)
    signed[(pred.abs() <= threshold)] = 0.0

    s[persisted] = signed[persisted]
    return s 


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

def regime_persistence_mask(
        regimes: pd.Series, 
        active_regime: int, 
        min_consecutive_days: int = 2,
    ) -> pd.Series:
    
    """
    Returns a boolean mask - true on dates where active_regime has persisted
    for >= min_consecutive_days ending at that date
    """

    reg = regimes.astype(int)
    is_active = (reg == int(active_regime)).astype(int)

    # rolling sum over consecutive days - True if last N days were active (sum == N)

    persisted = is_active.rolling(min_consecutive_days).sum() == min_consecutive_days
    persisted = persisted.fillna(False)
    persisted.name = "regime_persisted"
    return persisted

def vol_scaled_weights(
    signal: pd.Series,
    vol: pd.Series, 
    target_vol: float = 0.01, # 1% daily vol target
    max_leverage: float = 2.0,
    eps: float = 1e-12, 
) -> pd.Series:
    
    """
    Convert {-1, 0, +1} signals into volatility-scaled weights
    weight_t = signal_t * target_vol / vol_t, capped at max leverage
    """

    s = signal.astype(float)
    v = vol.astype(float)

    raw = s * (target_vol / (v + eps))
    capped = raw.clip(lower=-max_leverage, upper=max_leverage)
    capped.name = "weight"
    return capped 
    

