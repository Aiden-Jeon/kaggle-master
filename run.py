"""고정 CV fold 로 모델을 돌려 OOF/테스트 예측과 점수를 저장.

예: uv run python run.py spaceship-titanic lgbm catboost tabpfn_v2 tabicl
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from core.competition import ROOT, TRANSFORMS, get_folds, load_competition, load_data, load_features
from core.metrics import EXTRA, score
from models import make_model

RESULTS = ROOT / "results"


def predict(model, X, task):
    return model.predict(X) if task == "regression" else model.predict_proba(X)


def run_model(name, comp, X, y, X_test, cat_cols, folds, max_folds=None, no_test=False):
    n_splits = folds.max() + 1
    run_folds = range(n_splits if max_folds is None else max_folds)
    n_out = () if comp.task == "regression" else (len(np.unique(y)),)
    oof = np.full((len(y), *n_out), np.nan)
    test_pred = np.zeros((len(X_test), *n_out))
    fold_scores, t0 = [], time.time()

    for f in run_folds:
        tr, va = folds != f, folds == f
        model = make_model(name, comp.task, cat_cols, seed=comp.cv["seed"])
        if hasattr(model, "max_rows"):
            model.max_rows = comp.tfm_max_rows
        model.fit(X[tr], y[tr])
        oof[va] = predict(model, X[va], comp.task)
        if not no_test:
            test_pred += predict(model, X_test, comp.task) / len(run_folds)
        fold_scores.append(score(comp.metric, y[va], oof[va]))
        print(f"  [{name}] fold {f}: {comp.metric}={fold_scores[-1]:.5f}  ({time.time() - t0:.0f}s)")

    mask = ~np.isnan(oof).reshape(len(y), -1).any(1)
    scores = {m: score(m, y[mask], oof[mask]) for m in dict.fromkeys([comp.metric, *EXTRA[comp.task]])}
    return oof, test_pred, scores, fold_scores, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("comp")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--max-folds", type=int, default=None, help="빠른 확인용: 앞쪽 n개 fold만 실행")
    ap.add_argument("--note", default="", help="results.csv 에 남길 메모")
    ap.add_argument("--no-test", action="store_true", help="실험용: 테스트 예측 생략 (test.npy 저장 안 함)")
    args = ap.parse_args()

    comp = load_competition(args.comp)
    train, test = load_data(comp)
    X, X_test, cat_cols = load_features(comp).build(train, test)

    y_raw = train[comp.target]
    classes = None
    if comp.is_classification:
        enc = LabelEncoder().fit(y_raw)
        y, classes = enc.transform(y_raw), enc.classes_.tolist()
    else:
        y = y_raw.to_numpy(dtype=float)
        if comp.target_transform:
            y = TRANSFORMS[comp.target_transform][0](y)
    folds = get_folds(comp, y)
    print(f"{comp.name}: train {X.shape}, test {X_test.shape}, cats={cat_cols}")

    failed = []
    for name in args.models:
        try:
            oof, test_pred, scores, fold_scores, secs = run_model(name, comp, X, y, X_test, cat_cols, folds,
                                                                  args.max_folds, args.no_test)
        except Exception:  # 한 모델이 실패해도 나머지는 계속 실행
            traceback.print_exc()
            failed.append(name)
            print(f"=> {name}: 실패\n")
            continue
        out = RESULTS / comp.name / name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "oof.npy", oof)
        if not args.no_test:
            np.save(out / "test.npy", test_pred)
        meta = dict(comp=comp.name, model=name, metric=comp.metric, scores=scores, fold_scores=fold_scores,
                    seconds=round(secs, 1), classes=classes, n_folds=len(fold_scores), note=args.note,
                    timestamp=datetime.now().isoformat(timespec="seconds"))
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

        row = dict(timestamp=meta["timestamp"], comp=comp.name, model=name, metric=comp.metric,
                   cv=scores[comp.metric], cv_std=float(np.std(fold_scores)), n_folds=len(fold_scores),
                   seconds=meta["seconds"], **{f"cv_{k}": v for k, v in scores.items() if k != comp.metric},
                   note=args.note)
        csv = RESULTS / "results.csv"
        pd.concat([pd.read_csv(csv), pd.DataFrame([row])] if csv.exists() else [pd.DataFrame([row])]) \
            .to_csv(csv, index=False)
        print(f"=> {name}: {comp.metric}={scores[comp.metric]:.5f} ± {row['cv_std']:.5f}  ({secs:.0f}s)\n")

    if failed:
        print(f"실패한 모델: {', '.join(failed)}")


if __name__ == "__main__":
    main()
