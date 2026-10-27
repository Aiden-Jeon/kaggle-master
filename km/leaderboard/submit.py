"""저장된 테스트 예측으로 제출 파일 생성 (+ 선택적으로 Kaggle 제출).

예: uv run km submit spaceship-titanic catboost
    uv run km submit spaceship-titanic catboost tabpfn_v2 --push -m "cb+tabpfn 평균"
여러 모델을 주면 예측을 단순 평균.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from km.core.competition import TRANSFORMS, load_competition, load_data
from km.core.results import RESULTS


def make_submission(comp_name: str, models: list[str]) -> Path:
    """모델들의 test.npy 평균으로 제출 파일을 만들어 경로를 반환."""
    comp = load_competition(comp_name)
    _, test = load_data(comp)
    pred = np.mean([np.load(RESULTS / comp.name / m / "test.npy") for m in models], axis=0)

    ids = test[comp.id_col] if comp.id_col else np.arange(1, len(test) + 1)
    id_name, target_name = comp.submit_columns or [comp.id_col, comp.target]
    if comp.is_classification:
        classes = json.loads((RESULTS / comp.name / models[0] / "meta.json").read_text())["classes"]
    if comp.task == "multiclass" and comp.submit_proba:  # 클래스마다 확률 열 하나 (열 이름 = 클래스 값)
        sub = pd.concat([pd.DataFrame({id_name: ids}), pd.DataFrame(pred, columns=classes)], axis=1)
    else:
        if comp.is_classification and comp.submit_proba:
            target = pred[:, 1]
        elif comp.is_classification:
            target = np.asarray(classes, dtype=object)[pred.argmax(1)]
        else:
            target = TRANSFORMS[comp.target_transform][1](pred) if comp.target_transform else pred
        sub = pd.DataFrame({id_name: ids, target_name: target})

    path = RESULTS / comp.name / "submissions" / f"{'+'.join(models)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"저장: {path}\n{sub.head()}")
    return path


# 제출 설명: `<모델> | <n>-fold CV <점수> | run <실행 시각>`. 실행 시각 = 제출하는 test.npy 를 만든 실행의 meta.json timestamp
# (= results.csv 의 timestamp, UTC) → km lb 가 제출을 그 실행에 정확히 연결. 쓰는 곳(run_message)과 읽는 곳(parse_message)은 여기만.
_CV = re.compile(r"(\d+)-fold CV (-?\d+(?:\.\d+)?)(?!\.?\d)")  # `1.2.3` 같은 깨진 숫자는 무시
_RUN = re.compile(r"\brun (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)")


@dataclass
class SubmissionInfo:
    n_folds: int | None = None
    cv: str | None = None  # 설명에 적힌 그대로 (소수 5자리 문자열)
    run: str | None = None


def test_run(comp, model: str) -> dict | None:
    """저장된 test.npy 를 만든 실행 {run, cv, n_folds} (meta.json 의 test. 2026-10-27 전 meta 는 그 meta 의 실행으로 봄)."""
    meta = RESULTS / comp.name / model / "meta.json"
    if not meta.exists():
        return None
    m = json.loads(meta.read_text())
    if "test" in m:
        return m["test"]
    return dict(run=m["timestamp"], cv=m["scores"][comp.metric], n_folds=m["n_folds"])


def run_message(comp, model: str) -> str | None:
    """단일 모델 제출 설명의 `<n>-fold CV <점수> | run <시각>` 부분. 저장된 예측이 없으면 None."""
    t = test_run(comp, model)
    return f"{t['n_folds']}-fold CV {t['cv']:.5f} | run {t['run']}" if t else None


def parse_message(description: str | None) -> SubmissionInfo:
    text = description or ""
    cv, run = _CV.search(text), _RUN.search(text)
    return SubmissionInfo(int(cv.group(1)) if cv else None, cv.group(2) if cv else None, run.group(1) if run else None)


def push(comp_name: str, path: Path, name: str, message: str | None = None):
    """설명 맨 앞에 모델 이름 → `km lb` 가 제출을 모델에 연결. message 가 없으면 run_message (앙상블은 이름만)."""
    if message is None and "+" not in name:
        message = run_message(load_competition(comp_name), name)
    subprocess.run(["kaggle", "competitions", "submit", "-c", comp_name, "-f", str(path),
                    "-m", f"{name} | {message}" if message else name], check=True)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km submit")
    ap.add_argument("comp")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--push", action="store_true", help="kaggle 로 실제 제출")
    ap.add_argument("-m", "--message", default=None)
    args = ap.parse_args(argv)

    path = make_submission(args.comp, args.models)
    if args.push:
        push(args.comp, path, "+".join(args.models), args.message)

