import pandas as pd 
import numpy as np

def build_state_features(close: pd.Series) -> pd.DataFrame:
    r = np.log(close).diff()

    X = pd.DataFrame(index=close.index)

    # Returns
    X["r_1"] = r.shift(1)
    X["r_5"] = r.shift(1).rolling(5).sum()
    X["r_20"] = r.shift(1).rolling(20).sum()

    # Volatility
    X["vol_5"] = r.shift(1).rolling(5).std()
    X["vol_20"] = r.shift(1).rolling(20).std()

    # Trend
    X["trend_20"] = close / close.rolling(20).mean() - 1
    X["trend_60"] = close / close.rolling(60).mean() - 1

    # Drawdown 
    roll_max = close.rolling(252).max()
    X["dd_252"] = close / roll_max - 1

    return X.dropna()


