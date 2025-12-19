from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

# Project root = folder containing this file’s grandparent: timeserieslab/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class LabSettings:
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"

SETTINGS = LabSettings()
