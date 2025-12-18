from __future__ import annotations
import pandas as pd 

def load_ohlc_data(path: str) -> pd.Series:
    df = pd.read_csv(path) 
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("CSV must contain columns: 'date' and 'close'")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    close = df["close"].astype(float)
    close.name = "close"
    return close

