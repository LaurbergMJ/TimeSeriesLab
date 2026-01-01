from __future__ import annotations
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def fit_state_scaler(X: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X.values)
    return scaler 

def transform_states(
        X: pd.DataFrame, 
        scaler: StandardScaler
) -> pd.DataFrame:
    Z = scaler.transform(X.values)
    return pd.DataFrame(Z, index=X.index, columns=X.columns)

def fit_knn(
        Z: pd.DataFrame,
        n_neighbors: int = 20,
        metric: str = "euclidean",
) -> NearestNeighbors:
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
    )
    nn.fit(Z.values)
    return nn 

def query_analogs(
        t: pd.Timestamp,
        Z: pd.DataFrame,
        nn: NearestNeighbors,
        min_lookback_days: int = 252,
        k: int = 20,
) -> pd.Index:
    
    """
    Returns indices of k most similar historical states to date t, 
    excluding the last min_lookback_days
    """

    if t not in Z.index:
        raise KeyError(f"{t} not in state index")
    
    # eligible past states
    eligible = Z.loc[Z.index <= t - pd.Timedelta(days=min_lookback_days)]

    if len(eligible) < k:
        return pd.Index([])
    
    # query on full Z, then filter
    distances, indices = nn.kneighbors(
        Z.loc[[t]].values, 
        n_neighbors=len(Z)
    )

    neighbor_dates = Z.index[indices[0]]
    valid = [d for d in neighbor_dates if d in eligible.index]
    return pd.Index(valid[:k])


