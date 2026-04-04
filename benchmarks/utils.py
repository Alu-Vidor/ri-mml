from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = ROOT / "benchmarks"
DATA_DIR = BENCHMARKS_DIR / "data"
CONFIGS_DIR = BENCHMARKS_DIR / "configs"
DIGITIZED_DIR = BENCHMARKS_DIR / "digitized"
RESULTS_DIR = BENCHMARKS_DIR / "results"
REPORT_DIR = BENCHMARKS_DIR / "report"


def ensure_directories() -> None:
    for directory in (
        BENCHMARKS_DIR,
        DATA_DIR,
        CONFIGS_DIR,
        DIGITIZED_DIR,
        RESULTS_DIR,
        REPORT_DIR,
        RESULTS_DIR / "plots",
        RESULTS_DIR / "tables",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload):
        payload = asdict(payload)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

