from __future__ import annotations

import numpy as np 
import pandas as pd 

def forward_returns(
    close: pd.Series, 
    start_dates: pd.Index, 
    horizon: int = 5, 
) -> pd.Series:
    
    """
    Forward log return from t+1 to t+horizon for each start date
    """

    r = np.log(close).diff()
    out = {}

    for d in start_dates:
        try:
            window = r.loc[d:].iloc[1 : horizon + 1]
            out[d] = window.sum()
        except Exception:
            continue

    return pd.Series(out, name=f"fwd_ret_{horizon}")

def forward_volatility(
    close: pd.Series,
    start_dates: pd.Index,
    horizon: int = 5,
    ) -> pd.Series:
    
    """
    Realized volatility over t+1... t+horizon
    """
    r = np.log(close).diff()
    out = {}

    for d in start_dates:
        try:
            window=r.loc[d:].iloc[1:horizon + 1]
            out[d] = window.std()
        except Exception:
            continue

    return pd.Series(out, name=f"fwd_vol_{horizon}")
