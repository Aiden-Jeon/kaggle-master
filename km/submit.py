"""저장된 테스트 예측으로 제출 파일 생성 (+ 선택적으로 Kaggle 제출).

예: uv run km submit spaceship-titanic catboost
    uv run km submit spaceship-titanic catboost tabpfn_v2 --push -m "cb+tabpfn 평균"
여러 모델을 주면 예측을 단순 평균.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from km.competition import TRANSFORMS, load_competition, load_data
from km.results import RESULTS


def make_submission(comp_name: str, models: list[str]) -> Path:
    """모델들의 test.npy 평균으로 제출 파일을 만들어 경로를 반환."""
    comp = load_competition(comp_name)
    _, test = load_data(comp)
    pred = np.mean([np.load(RESULTS / comp.name / m / "test.npy") for m in models], axis=0)

    if comp.is_classification and comp.submit_proba:
        target = pred[:, 1]
    elif comp.is_classification:
        classes = json.loads((RESULTS / comp.name / models[0] / "meta.json").read_text())["classes"]
        target = np.asarray(classes, dtype=object)[pred.argmax(1)]
    else:
        target = TRANSFORMS[comp.target_transform][1](pred) if comp.target_transform else pred
    ids = test[comp.id_col] if comp.id_col else np.arange(1, len(test) + 1)
    id_name, target_name = comp.submit_columns or [comp.id_col, comp.target]
    sub = pd.DataFrame({id_name: ids, target_name: target})

    path = RESULTS / comp.name / "submissions" / f"{'+'.join(models)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"저장: {path}\n{sub.head()}")
    return path


def push(comp_name: str, path: Path, name: str, message: str | None = None):
    """설명 맨 앞에 모델 이름 → `km lb` 가 제출을 모델에 연결."""
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

