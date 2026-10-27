"""실험 결과 위치와 누적 결과표(results/results.csv)."""

from __future__ import annotations

import fcntl
import shutil
from pathlib import Path

import pandas as pd

from km.core.competition import ROOT

RESULTS = ROOT / "results"


def append_rows(rows: pd.DataFrame):
    """results.csv 에 행 추가. 여러 작업자가 동시에 쓰므로 잠금을 걸고 씀."""
    csv = RESULTS / "results.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / ".results.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pd.concat([pd.read_csv(csv), rows] if csv.exists() else [rows]).to_csv(csv, index=False)


def full_runs(comp) -> pd.DataFrame:
    """results.csv 에서 대회의 전체 fold 실행들, 시각 순 (timestamp 는 UTC)."""
    csv = RESULTS / "results.csv"
    if not csv.exists():
        return pd.DataFrame(columns=["timestamp", "model", "cv"])
    r = pd.read_csv(csv)
    return r[(r.comp == comp.name) & (r.n_folds == comp.cv["n_splits"])].sort_values("timestamp")



def merge_results(remote: Path, where: str):
    """원격 실행의 results/ 를 로컬 results/ 에 병합 (results.csv 는 where 열을 붙여 행 추가)."""
    for d in remote.iterdir():
        if d.is_dir():
            shutil.copytree(d, RESULTS / d.name, dirs_exist_ok=True)
    rows = pd.read_csv(remote / "results.csv").assign(where=where)
    append_rows(rows)
    print(rows.to_string(index=False))
