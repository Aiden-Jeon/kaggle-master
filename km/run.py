"""고정 CV fold 로 모델을 돌려 OOF/테스트 예측과 점수를 저장.

예: uv run km run spaceship-titanic lgbm catboost tabpfn_v2 tabicl
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from km.competition import TRANSFORMS, get_folds, load_competition, load_data, load_features
from km.remote.common import add_run_args, run_argv
from km.metrics import EXTRA, score
from km.results import RESULTS, append_rows
from km.models import make_model


def predict(model, X, task):
    return model.predict(X) if task == "regression" else model.predict_proba(X)


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    return sum(line.startswith("GPU ") for line in out.splitlines())


def run_parallel(args: argparse.Namespace, n_gpus: int) -> list[str]:
    """GPU 마다 작업자 하나. 각 작업자가 남은 모델을 하나씩 가져가 자기 GPU 에서 `km run` 을 실행."""
    todo: queue.Queue[str] = queue.Queue()
    for m in args.models:
        todo.put(m)
    failed: list[str] = []

    def worker(gpu: int):
        while True:
            try:
                name = todo.get_nowait()
            except queue.Empty:
                return
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "KM_WORKER": "1"}
            proc = subprocess.Popen([sys.executable, "-u", "-m", "km", "run", args.comp, *run_argv(args, [name])],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:  # 어느 GPU 출력인지 접두어로 구분
                print(f"[gpu{gpu}] {line}", end="", flush=True)
            if proc.wait() != 0:
                failed.append(name)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(min(n_gpus, len(args.models)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failed


def run_model(name, comp, X, y, X_test, cat_cols, folds, max_folds=None, no_test=False):
    n_splits = folds.max() + 1
    run_folds = range(n_splits if max_folds is None else max_folds)
    n_out = () if comp.task == "regression" else (len(np.unique(y)),)
    oof = np.full((len(y), *n_out), np.nan)
    test_pred = np.zeros((len(X_test), *n_out))
    fold_scores, t0 = [], time.time()

    for f in run_folds:
        tr, va = folds != f, folds == f
        model = make_model(name, comp.task, cat_cols, seed=comp.cv["seed"], max_rows=comp.tfm_max_rows)
        model.fit(X[tr], y[tr])
        oof[va] = predict(model, X[va], comp.task)
        if not no_test:
            test_pred += predict(model, X_test, comp.task) / len(run_folds)
        fold_scores.append(score(comp.metric, y[va], oof[va]))
        print(f"  [{name}] fold {f}: {comp.metric}={fold_scores[-1]:.5f}  ({time.time() - t0:.0f}s)")

    mask = ~np.isnan(oof).reshape(len(y), -1).any(1)
    scores = {m: score(m, y[mask], oof[mask]) for m in dict.fromkeys([comp.metric, *EXTRA[comp.task]])}
    return oof, test_pred, scores, fold_scores, time.time() - t0


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km run")
    add_run_args(ap)
    ap.add_argument("--gpus", type=int, default=None, help="동시에 쓸 GPU 수 (기본: 감지된 전부)")
    args = ap.parse_args(argv)

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
    folds = get_folds(comp, y)  # 병렬 실행 전에 만들어 두어 모든 작업자가 같은 fold 사용
    print(f"{comp.name}: train {X.shape}, test {X_test.shape}, cats={cat_cols}")

    n_gpus = gpu_count() if args.gpus is None else args.gpus
    if n_gpus > 1 and len(args.models) > 1 and not os.environ.get("KM_WORKER"):
        print(f"GPU {n_gpus}장에서 모델 {len(args.models)}개 병렬 실행")
        failed = run_parallel(args, n_gpus)
        if failed:
            print(f"실패한 모델: {', '.join(failed)}")
        return

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
        append_rows(pd.DataFrame([row]))
        print(f"=> {name}: {comp.metric}={scores[comp.metric]:.5f} ± {row['cv_std']:.5f}  ({secs:.0f}s)\n")

    if failed:
        print(f"실패한 모델: {', '.join(failed)}")
        if os.environ.get("KM_WORKER"):
            sys.exit(1)  # 병렬 실행의 부모가 실패를 집계하도록

