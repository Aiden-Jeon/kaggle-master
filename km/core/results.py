"""실험 결과 위치와 누적 결과표(results/results.csv)."""

from __future__ import annotations

import fcntl
import json
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


TEST_INFO = "test.json"  # test.npy 옆: 그 예측을 만든 실행 {run, cv, n_folds}


def full_runs(comp) -> pd.DataFrame:
    """results.csv 에서 대회의 전체 fold 실행들, 시각 순 (timestamp 는 UTC)."""
    csv = RESULTS / "results.csv"
    if not csv.exists():
        return pd.DataFrame(columns=["timestamp", "model", "cv", "note", "test"])
    r = pd.read_csv(csv)
    return r[(r.comp == comp.name) & (r.n_folds == comp.cv["n_splits"])].sort_values("timestamp")


def made_test(runs: pd.DataFrame) -> pd.Series:
    """테스트 예측을 만든 실행인지. test 열(2026-10-27~)이 없던 예전 행은 `ctx ` 탐색 실험(테스트 예측 생략)만 아니라고 봄."""
    test = runs["test"] if "test" in runs else pd.Series(None, index=runs.index, dtype=object)
    return test.fillna(~runs.note.fillna("").str.startswith("ctx ")).astype(bool)


def write_test_info(out: Path, run: str, cv: float, n_folds: int):
    """test.npy 를 저장할 때만 같이 씀 → --no-test 재실행·원격 결과 병합(copytree)에도 예측과 출처가 함께 다님."""
    (out / TEST_INFO).write_text(json.dumps(dict(run=run, cv=cv, n_folds=n_folds), indent=2))


def test_info(comp_name: str, model: str) -> dict | None:
    """저장된 test.npy 를 만든 실행 {run, cv, n_folds}. test.json 이 없으면 None (추측하지 않음)."""
    d = RESULTS / comp_name / model
    return json.loads((d / TEST_INFO).read_text()) if (d / TEST_INFO).exists() and (d / "test.npy").exists() else None


def merge_results(remote: Path, where: str):
    """원격 실행의 results/ 를 로컬 results/ 에 병합 (results.csv 는 where 열을 붙여 행 추가)."""
    for d in remote.iterdir():
        if d.is_dir():
            shutil.copytree(d, RESULTS / d.name, dirs_exist_ok=True)
    rows = pd.read_csv(remote / "results.csv").assign(where=where)
    append_rows(rows)
    print(rows.to_string(index=False))
