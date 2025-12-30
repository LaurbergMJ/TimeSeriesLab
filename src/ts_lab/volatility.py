from __future__ import annotations
import numpy as np 
import pandas as pd 

def log_returns(close: pd.Series) -> pd.Series:
    
    r = np.log(close).diff()
    r.name = "log_return"
    return r

def realized_vol(
    close: pd.Series,
    window: int = 5, 
    annualize: bool = False,
    ) -> pd.Series:

    """
    Realized volatility computed from past returns over "window" days
    """

    r = log_returns(close)
    vol = r.rolling(window).std()

    if annualize: 
        vol = vol * np.sqrt(252)
    vol.name = f"rv_{window}"

    return vol 

def future_realized_vol_target(
    close: pd.Series, 
    window: int = 5,
    annualize: bool = False
) -> pd.Series:
    
    """
    Target at time t is realized vol over the next 'window' days
    - computed as realized_vol shifted backward by 'window' days
    """

    rv = realized_vol(close, window=window, annualize=annualize)
    y = rv.shift(-window)
    y.name = f"target_rv_next_{window}"
    return y


    