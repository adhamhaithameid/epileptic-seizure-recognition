from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class RunnerIO:
    metrics_csv: Path
    manifest_json: Path
    checkpoint_every: int = 100


@dataclass
class ResumeState:
    completed_rows: int = 0
    checkpoint_index: int = 0


def load_resume_state(io: RunnerIO) -> ResumeState:
    if not io.metrics_csv.exists():
        return ResumeState()

    df = pd.read_csv(io.metrics_csv)
    checkpoint_index = int(len(df))

    return ResumeState(
        completed_rows=int(len(df)),
        checkpoint_index=checkpoint_index,
    )


def write_manifest(io: RunnerIO, payload: Dict) -> None:
    io.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with io.manifest_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_checkpoint(io: RunnerIO, df: pd.DataFrame, overwrite: bool = False) -> None:
    io.metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not io.metrics_csv.exists():
        df.to_csv(io.metrics_csv, index=False)
        return

    df.to_csv(io.metrics_csv, mode="a", index=False, header=False)
