from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Paths:
    data_raw: Path = PROJECT_ROOT / "data" / "eod_SX5E_index"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    models: Path = PROJECT_ROOT / "models"

PATHS = Paths()

