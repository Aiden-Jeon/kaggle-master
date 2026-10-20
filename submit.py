"""저장된 테스트 예측으로 제출 파일 생성 (+ 선택적으로 Kaggle 제출).

예: uv run python submit.py spaceship-titanic catboost
    uv run python submit.py spaceship-titanic catboost tabpfn_v2 --push -m "cb+tabpfn 평균"
여러 모델을 주면 예측을 단순 평균.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import pandas as pd

from core.competition import TRANSFORMS, load_competition, load_data
from run import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("comp")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--push", action="store_true", help="kaggle 로 실제 제출")
    ap.add_argument("-m", "--message", default=None)
    args = ap.parse_args()

    comp = load_competition(args.comp)
    _, test = load_data(comp)
    pred = np.mean([np.load(RESULTS / comp.name / m / "test.npy") for m in args.models], axis=0)

    if comp.is_classification and comp.submit_proba:
        target = pred[:, 1]
    elif comp.is_classification:
        classes = json.loads((RESULTS / comp.name / args.models[0] / "meta.json").read_text())["classes"]
        target = np.asarray(classes, dtype=object)[pred.argmax(1)]
    else:
        target = TRANSFORMS[comp.target_transform][1](pred) if comp.target_transform else pred
    sub = pd.DataFrame({comp.id_col: test[comp.id_col], comp.target: target})

    name = "+".join(args.models)
    path = RESULTS / comp.name / "submissions" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"저장: {path}\n{sub.head()}")

    if args.push:
        subprocess.run(["kaggle", "competitions", "submit", "-c", comp.name, "-f", str(path),
                        "-m", args.message or name], check=True)


if __name__ == "__main__":
    main()
