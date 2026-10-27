"""실험 결과 위치와 누적 결과표(results/results.csv)."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from km.core.competition import ROOT


def outputs_dir() -> Path:
    """실행 출력 (git 제외: <대회>/<모델>/ 예측·OOF·test.json, submissions/, _air/runs.json) 위치.
    worktree 에서도 메인 체크아웃의 results/ → worktree 를 지워도 남음 (#22: worktree 에서 받은 Kaggle 예측 37개가 사라짐).
    KM_OUTPUTS 가 있으면 그 경로. 메인 체크아웃·원격 실행 (코드 묶음이라 .git 이 없음) 은 ROOT/results."""
    if env := os.environ.get("KM_OUTPUTS"):
        return Path(env).expanduser().resolve()
    if not (ROOT / ".git").is_file():  # worktree·submodule 만 .git 이 파일 ('gitdir: …')
        return ROOT / "results"
    try:
        common = Path(subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                     capture_output=True, text=True, check=True, timeout=30).stdout.strip())
    except (OSError, subprocess.SubprocessError) as e:  # 조용히 worktree 에 쓰면 #22 가 다시 생김 → 멈춤
        raise SystemExit(f"메인 체크아웃을 찾지 못함 ({e!r}). KM_OUTPUTS 로 실행 출력 위치를 지정") from e
    # submodule·bare 저장소의 worktree 는 메인 체크아웃이 없음 → 이 체크아웃의 results/
    return common.parent / "results" if common.name == ".git" else ROOT / "results"


RESULTS = ROOT / "results"  # git 추적 표 (results.csv, leaderboard.csv) 와 그 잠금: 체크아웃마다 → PR 로 메인에 들어감
OUTPUTS = outputs_dir()  # 실행 출력. 원격 실행·메인 체크아웃에서는 RESULTS 와 같음
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


def read_results() -> pd.DataFrame:
    """results.csv 전체. km lb 는 한 번만 읽고 대회마다 넘김."""
    csv = RESULTS / "results.csv"
    if not csv.exists():
        return pd.DataFrame(columns=["timestamp", "comp", "model", "cv", "n_folds", "note", "test"])
    return pd.read_csv(csv)


def parse_utc(text) -> datetime:
    """시각 문자열 → UTC 기준 naive datetime. 문자열 비교 대신 이걸로 비교.
    타임존이 없으면 UTC 로 봄: results.csv timestamp·test.json run 은 datetime.now(timezone.utc) 로 쓰고,
    Kaggle 제출 날짜는 kagglesdk 가 UTC 'Z' 를 떼고 naive 로 준 값 (2026-10-27 확인). 타임존이 붙어 있으면 UTC 로 환산.
    못 읽으면 RuntimeError (조용히 틀리지 않도록 그 대회·모델 처리를 멈춤)."""
    try:
        t = datetime.fromisoformat(str(text).strip())
    except ValueError as e:
        raise RuntimeError(f"시각을 읽지 못함: {text!r}") from e
    return t.astimezone(timezone.utc).replace(tzinfo=None) if t.tzinfo else t


TEST_INFO = "test.json"  # test.npy 옆: 그 예측을 만든 실행 {run, cv, n_folds}


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
    d = OUTPUTS / comp_name / model
    if not ((d / TEST_INFO).exists() and (d / "test.npy").exists()):
        return None
    try:
        info = json.loads((d / TEST_INFO).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return info if isinstance(info, dict) and {"run", "cv", "n_folds"} <= info.keys() else None


def merge_results(remote: Path, where: str):
    """원격 실행의 results/ 를 병합: 예측 등 실행 출력은 OUTPUTS 에 복사, results.csv 는 where 열을 붙여 이 체크아웃의 표에 행 추가."""
    for d in remote.iterdir():
        if d.is_dir():
            # test.json 없이 test.npy 만 온 모델 (test.json 을 안 쓰던 코드로 돌린 원격 실행) → 로컬의 예전 출처를 지움
            for m in d.iterdir():
                if (m / "test.npy").exists() and not (m / TEST_INFO).exists():
                    clear_test_info(OUTPUTS / d.name / m.name)
            shutil.copytree(d, OUTPUTS / d.name, dirs_exist_ok=True)
    if OUTPUTS != RESULTS:
        print(f"실행 출력 저장: {OUTPUTS} (results.csv 는 {RESULTS})")
    if not (remote / "results.csv").exists():  # --fold 실행은 results.csv 를 안 씀 (합칠 때 씀)
        return
    rows = pd.read_csv(remote / "results.csv").assign(where=where)
    append_rows(rows)
    print(rows.to_string(index=False))
