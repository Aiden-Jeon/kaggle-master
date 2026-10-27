"""고정 CV fold 로 모델을 돌려 OOF/테스트 예측과 점수를 저장.

예: uv run km run spaceship-titanic lgbm catboost tabpfn_v2 tabicl
fold 나눠 실행: 같은 명령에 --fold 0 … --fold 4 (따로따로 실행) → --merge-folds 로 합쳐 results.csv 에 한 줄
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from km.core.competition import TRANSFORMS, get_folds, load_competition, load_data, load_features
from km.core.metrics import EXTRA, score
from km.core.results import RESULTS, append_rows, clear_test_info, write_test_info
from km.models import make_model
from km.runner.common import add_run_args, run_argv


def predict(model, X, task):
    return model.predict(X) if task == "regression" else model.predict_proba(X)


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    return sum(line.startswith("GPU ") for line in out.splitlines())


def cpu_count() -> int:
    """이 프로세스가 쓸 수 있는 CPU 수 (affinity, cgroup v2 할당량 반영)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return n


def run_parallel(args: argparse.Namespace, n_gpus: int) -> list[str]:
    """GPU 마다 작업자 하나. 각 작업자가 남은 모델을 하나씩 가져가 자기 GPU 에서 `km run` 을 실행.

    작업자끼리 CPU 를 나눠 씀 → 스레드 수를 CPU 수 / 작업자 수로 제한 (OMP_NUM_THREADS, GBDT 는 KM_THREADS).
    제한이 없으면 lgbm·xgb 가 각자 전체 코어만큼 스레드를 띄워 극도로 느려짐 (8xH100: 110행 xgb fold 하나 593s).
    """
    todo: queue.Queue[str] = queue.Queue()
    for m in args.models:
        todo.put(m)
    failed: list[str] = []
    n_workers = min(n_gpus, len(args.models))
    threads_per = str(max(1, cpu_count() // n_workers))
    print(f"작업자 {n_workers}개, 작업자당 CPU 스레드 {threads_per}")

    def worker(gpu: int):
        while True:
            try:
                name = todo.get_nowait()
            except queue.Empty:
                return
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "KM_WORKER": "1",
                   "OMP_NUM_THREADS": threads_per, "KM_THREADS": threads_per}
            proc = subprocess.Popen([sys.executable, "-u", "-m", "km", "run", args.comp, *run_argv(args, [name])],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:  # 어느 GPU 출력인지 접두어로 구분
                print(f"[gpu{gpu}] {line}", end="", flush=True)
            if proc.wait() != 0:
                failed.append(name)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failed


def run_model(name, comp, X, y, X_test, cat_cols, folds, max_folds=None, no_test=False, classes=None, fold=None):
    n_splits = folds.max() + 1
    run_folds = [fold] if fold is not None else range(n_splits if max_folds is None else max_folds)
    n_out = () if comp.task == "regression" else (len(np.unique(y)),)
    oof = np.full((len(y), *n_out), np.nan)
    test_pred = np.zeros((len(X_test), *n_out))
    fold_scores, t0 = [], time.time()

    for f in run_folds:
        tr, va = folds != f, folds == f
        model = make_model(name, comp.task, cat_cols, seed=comp.cv["seed"], max_rows=comp.tfm_max_rows,
                           max_features=comp.tfm_max_features, prompt=comp.llm_prompt(classes), metric=comp.metric)
        t_fit = time.time()
        model.fit(X[tr], y[tr])
        t_pred = time.time()
        # 검증 fold 와 테스트를 한 번에 예측 → TFM 이 컨텍스트를 fold 당 한 번만 처리
        if no_test:
            oof[va] = predict(model, X[va], comp.task)
        elif getattr(model, "joint_predict", True):
            pred = predict(model, pd.concat([X[va], X_test]), comp.task)
            oof[va], test = pred[:va.sum()], pred[va.sum():]
        else:
            oof[va], test = predict(model, X[va], comp.task), predict(model, X_test, comp.task)
        if not no_test:
            test_pred += test / len(run_folds)
        fold_scores.append(score(comp.metric, y[va], oof[va]))
        print(f"  [{name}] fold {f}: {comp.metric}={fold_scores[-1]:.5f}  "
              f"(fit {t_pred - t_fit:.0f}s, pred {time.time() - t_pred:.0f}s, 누적 {time.time() - t0:.0f}s)")

    mask = ~np.isnan(oof).reshape(len(y), -1).any(1)
    scores = {m: score(m, y[mask], oof[mask]) for m in dict.fromkeys([comp.metric, *EXTRA[comp.task]])}
    return oof, test_pred, scores, fold_scores, time.time() - t0


def target(comp, train):
    """학습 타깃 (분류는 0..n-1 인코딩 + 원래 클래스, 회귀는 target_transform 적용)."""
    y_raw = train[comp.target]
    if comp.is_classification:
        enc = LabelEncoder().fit(y_raw)
        return enc.transform(y_raw), enc.classes_.tolist()
    y = y_raw.to_numpy(dtype=float)
    return (TRANSFORMS[comp.target_transform][0](y) if comp.target_transform else y), None


def utc_now() -> str:
    # UTC (Kaggle 제출 시각과 비교하므로). 2026-10-22 로컬 실행 4행은 KST 였다가 UTC 로 고침
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def save_run(comp, name, oof, test_pred, scores, fold_scores, secs, classes, note, no_test, where=None):
    """전체 실행 결과: results/<대회>/<모델>/{oof,test}.npy·meta.json·test.json + results.csv 한 줄."""
    out = RESULTS / comp.name / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof.npy", oof)
    meta = dict(comp=comp.name, model=name, metric=comp.metric, scores=scores, fold_scores=fold_scores,
                seconds=round(secs, 1), classes=classes, n_folds=len(fold_scores), note=note, timestamp=utc_now())
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    if not no_test:  # 예측과 그 출처(km submit 이 제출 설명에 적음)는 함께만 씀
        clear_test_info(out)
        np.save(out / "test.npy", test_pred)
        write_test_info(out, meta["timestamp"], scores[comp.metric], len(fold_scores), classes)

    row = dict(timestamp=meta["timestamp"], comp=comp.name, model=name, metric=comp.metric,
               cv=scores[comp.metric], cv_std=float(np.std(fold_scores)), n_folds=len(fold_scores),
               seconds=meta["seconds"], **{f"cv_{k}": v for k, v in scores.items() if k != comp.metric},
               note=note, test=not no_test, **({"where": where} if where else {}))
    append_rows(pd.DataFrame([row]))
    print(f"=> {name}: {comp.metric}={scores[comp.metric]:.5f} ± {row['cv_std']:.5f}  ({secs:.0f}s)\n")


def save_fold(comp, name, fold, oof, test_pred, fold_score, secs, classes, note, no_test):
    """--fold 실행 결과: results/<대회>/<모델>/folds/<k>/ (results.csv 에는 merge_folds 가 합친 뒤 씀)."""
    out = RESULTS / comp.name / name / "folds" / str(fold)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof.npy", oof)
    if not no_test:
        np.save(out / "test.npy", test_pred)
    meta = dict(fold=fold, fold_score=fold_score, seconds=round(secs, 1), classes=classes, note=note,
                no_test=no_test, timestamp=utc_now())
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"=> {name}: fold {fold} {comp.metric}={fold_score:.5f}  ({secs:.0f}s) → {out}\n")


def merge_folds(comp, name, y, where=None) -> bool:
    """folds/<k>/ 가 전부 있으면 합쳐 save_run (OOF 는 fold 별로 채우고 test 는 평균 = 한 번에 돌린 것과 같음).
    seconds 는 fold 시간 합. 모델 로드가 fold 마다 들어가 한 번에 돌린 것보다 길 수 있어 note 에 `(fold 분할)` 표시.
    합친 뒤 folds/ 는 지움. 빠진 fold 가 있으면 False."""
    d = RESULTS / comp.name / name / "folds"
    n = comp.cv["n_splits"]
    metas = {k: json.loads((d / str(k) / "meta.json").read_text())
             for k in range(n) if (d / str(k) / "meta.json").exists()}
    if missing := [k for k in range(n) if k not in metas]:
        print(f"{name}: fold {missing} 없음 → 합치지 않음 (--fold k 로 채운 뒤 --merge-folds)")
        return False
    for key in ("no_test", "classes", "note"):
        if len({json.dumps(m[key]) for m in metas.values()}) > 1:
            print(f"{name}: fold 마다 {key} 가 다름 → 합치지 않음 ({d})")
            return False
    first = metas[0]
    oof = np.load(d / "0" / "oof.npy")
    for k in range(1, n):
        o = np.load(d / str(k) / "oof.npy")
        rows = ~np.isnan(o).reshape(len(o), -1).any(1)
        oof[rows] = o[rows]
    test_pred = None if first["no_test"] else np.mean([np.load(d / str(k) / "test.npy") for k in range(n)], axis=0)
    scores = {m: score(m, y, oof) for m in dict.fromkeys([comp.metric, *EXTRA[comp.task]])}
    save_run(comp, name, oof, test_pred, scores, [metas[k]["fold_score"] for k in range(n)],
             sum(m["seconds"] for m in metas.values()), first["classes"], f"{first['note']} (fold 분할)".strip(),
             first["no_test"], where)
    shutil.rmtree(d)
    return True


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km run")
    add_run_args(ap)
    ap.add_argument("--gpus", type=int, default=None, help="동시에 쓸 GPU 수 (기본: 감지된 전부)")
    ap.add_argument("--merge-folds", action="store_true",
                    help="실행하지 않고 --fold 로 저장한 fold 들을 합쳐 results.csv 에 기록")
    ap.add_argument("--where", help="--merge-folds 로 쓰는 행의 실행 위치 (예: databricks)")
    args = ap.parse_args(argv)
    if args.fold is not None and args.max_folds:
        ap.error("--fold 와 --max-folds 는 같이 쓸 수 없음")

    comp = load_competition(args.comp)
    train, test = load_data(comp)
    y, classes = target(comp, train)
    if args.merge_folds:
        failed = [name for name in args.models if not merge_folds(comp, name, y, args.where)]
        if failed:
            sys.exit(f"합치지 못한 모델: {', '.join(failed)}")
        return
    if args.fold is not None and not 0 <= args.fold < comp.cv["n_splits"]:
        ap.error(f"--fold 는 0 ~ {comp.cv['n_splits'] - 1}")
    X, X_test, cat_cols = load_features(comp).build(train, test)
    folds = get_folds(comp, y)  # 병렬 실행 전에 만들어 두어 모든 작업자가 같은 fold 사용
    print(f"{comp.name}: train {X.shape}, test {X_test.shape}, cats={cat_cols}")

    n_gpus = gpu_count() if args.gpus is None else args.gpus
    # GPU 가 있으면 GPU 1장이어도 모델마다 새 프로세스 → 앞 모델의 GPU/RAM 이 다음 모델에 남지 않음
    if n_gpus >= 1 and len(args.models) > 1 and not os.environ.get("KM_WORKER"):
        print(f"GPU {n_gpus}장에서 모델 {len(args.models)}개를 모델별 프로세스로 실행")
        failed = run_parallel(args, n_gpus)
        if failed:
            print(f"실패한 모델: {', '.join(failed)}")
        return

    failed = []
    for name in args.models:
        try:
            oof, test_pred, scores, fold_scores, secs = run_model(name, comp, X, y, X_test, cat_cols, folds,
                                                                  args.max_folds, args.no_test, classes, args.fold)
        except Exception:  # 한 모델이 실패해도 나머지는 계속 실행
            traceback.print_exc()
            failed.append(name)
            print(f"=> {name}: 실패\n")
            continue
        if args.fold is not None:
            save_fold(comp, name, args.fold, oof, test_pred, fold_scores[0], secs, classes, args.note, args.no_test)
        else:
            save_run(comp, name, oof, test_pred, scores, fold_scores, secs, classes, args.note, args.no_test)

    if failed:
        print(f"실패한 모델: {', '.join(failed)}")
        if os.environ.get("KM_WORKER"):
            sys.exit(1)  # 병렬 실행의 부모가 실패를 집계하도록

