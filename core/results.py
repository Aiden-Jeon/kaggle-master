"""실험 결과 위치와 누적 결과표(results/results.csv)."""

from __future__ import annotations

import fcntl

import pandas as pd

from core.competition import ROOT

RESULTS = ROOT / "results"


def append_rows(rows: pd.DataFrame):
    """results.csv 에 행 추가. 여러 작업자가 동시에 쓰므로 잠금을 걸고 씀."""
    csv = RESULTS / "results.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / ".results.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pd.concat([pd.read_csv(csv), rows] if csv.exists() else [rows]).to_csv(csv, index=False)
