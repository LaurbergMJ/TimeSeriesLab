from __future__ import annotations
from pathlib import Path
import pandas as pd 

def list_csv_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.csv"))

def load_close_data(path: str) -> pd.Series:
    df = pd.read_csv(path) 

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{path.name}: expected columns ['date', 'close].")
    
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    close = df["close"].astype(float)
    close.name = "close"
    return close



