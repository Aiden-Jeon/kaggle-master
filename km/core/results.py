"""실험 결과 위치와 누적 결과표(results/results.csv)."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd

from km.core.competition import ROOT

RESULTS = ROOT / "results"
# 탐색 실행 (km board 에서 뺌): 컨텍스트 탐색(`ctx …`), 결정 모델 실험 중 같이 돌린 GBDT(`llm…`, `sap…`).
# 결정 모델(llm·sap) 실행도 테스트 예측이 있어 km lb 는 이 규칙을 쓰지 않음 (제출 ↔ 실행 연결이 끊김).
EXPLORATION = re.compile(r"ctx |llm|sap")
# 테스트 예측을 안 만든 실행의 note (test 열이 없던 예전 행의 추정, 백필 값). 1-fold 탐색(probe·ctx-size)도 포함.
NO_TEST = re.compile(r"ctx[ -]|probe")


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


def is_exploration(note: pd.Series) -> pd.Series:
    """탐색 실행인지 (km board 가 LB 표에서 뺌). note 가 비었으면 False."""
    return note.fillna("").str.match(EXPLORATION)


def made_test(runs: pd.DataFrame) -> pd.Series:
    """테스트 예측을 만든 실행인지 (km lb 의 제출 연결 후보, km board). test 열이 없거나 빈 값이면 note 로 추정 (NO_TEST)."""
    test = runs["test"] if "test" in runs else pd.Series(None, index=runs.index, dtype=object)
    return test.fillna(~runs.note.fillna("").str.match(NO_TEST)).astype(bool)


def clear_test_info(out: Path):
    """test.npy 를 새로 쓰기 전에 호출 → 저장 도중 멈추거나 test.json 을 안 쓰는 코드가 덮어써도 예전 출처가 남지 않음."""
    (out / TEST_INFO).unlink(missing_ok=True)


def save_test_info(out: Path, info: dict):
    """test.json 을 임시 파일 → 교체로 원자적으로 씀."""
    tmp = out / f".{TEST_INFO}.tmp"
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, out / TEST_INFO)


def write_test_info(out: Path, run: str, cv: float, n_folds: int, classes: list | None):
    """test.npy 를 저장한 직후에만 씀 → --no-test 재실행·원격 결과 병합(copytree)에도 예측과 출처가 함께 다님.
    classes: 예측 열 순서 (meta.json 은 --no-test 재실행이 덮어쓰므로 제출 파일은 이 값으로 만듦)."""
    save_test_info(out, dict(run=run, cv=cv, n_folds=n_folds, classes=classes))


def test_info(comp_name: str, model: str) -> dict | None:
    """저장된 test.npy 를 만든 실행 {run, cv, n_folds, classes}. test.json 이 없거나 깨졌으면 None (추측하지 않음)."""
    d = RESULTS / comp_name / model
    if not ((d / TEST_INFO).exists() and (d / "test.npy").exists()):
        return None
    try:
        info = json.loads((d / TEST_INFO).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return info if isinstance(info, dict) and {"run", "cv", "n_folds"} <= info.keys() else None


def merge_results(remote: Path, where: str):
    """원격 실행의 results/ 를 로컬 results/ 에 병합 (results.csv 는 where 열을 붙여 행 추가)."""
    for d in remote.iterdir():
        if d.is_dir():
            # test.json 없이 test.npy 만 온 모델 (test.json 을 안 쓰던 코드로 돌린 원격 실행) → 로컬의 예전 출처를 지움
            for m in d.iterdir():
                if (m / "test.npy").exists() and not (m / TEST_INFO).exists():
                    clear_test_info(RESULTS / d.name / m.name)
            shutil.copytree(d, RESULTS / d.name, dirs_exist_ok=True)
    rows = pd.read_csv(remote / "results.csv").assign(where=where)
    append_rows(rows)
    print(rows.to_string(index=False))
