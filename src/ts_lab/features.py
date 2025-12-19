from __future__ import annotations
import pandas as pd
import numpy as np

def compute_returns(close: pd.Series) -> pd.Series:
    r = np.log(close).diff()
    r.name = "log_returns"
    return r

def build_features(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:

    r = compute_returns(close)

    X = pd.DataFrame(index=close.index)
    X["r_1"] = r.shift(0)
    X["r_2"] = r.shift(1)
    X["r_3"] = r.shift(2)

    X["vol_5"] = r.rolling(5).std()
    X["vol_20"] = r.rolling(20).std()

    ma_20 = close.rolling(20).mean()
    X["trend_20"] = (close / ma_20) - 1.0

    y = r.shift(-1)
    y.name = "target_next_log_return"

    df = pd.concat([X, y], axis=1).dropna()

    return df.drop(columns=[y.name]), df[y.name]
