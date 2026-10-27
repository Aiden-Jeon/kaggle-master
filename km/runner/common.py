"""`km run` 을 다른 곳(GPU 별 작업자 프로세스, Kaggle·Colab·Databricks)에서 실행할 때 공유하는 인자와 코드 묶음."""

from __future__ import annotations

import argparse
import base64
import io
import tarfile

from km.core.competition import ROOT, load_competition
from km.models import GATED, KAGGLE_PIP, base_name

CODE_PATHS = ["km", "competitions"]  # 원격에서는 이 위치를 cwd 로 두고 `python -m km run` 실행


def add_run_args(ap: argparse.ArgumentParser, models_nargs: str = "+"):
    """`km run` 으로 그대로 전달되는 인자. 새 인자를 추가하면 run_argv 에도 반영."""
    ap.add_argument("comp")
    ap.add_argument("models", nargs=models_nargs)
    ap.add_argument("--max-folds", type=int, default=None, help="빠른 확인용: 앞쪽 n개 fold만 실행")
    ap.add_argument("--note", default="", help="results.csv 에 남길 메모")
    ap.add_argument("--no-test", action="store_true", help="실험용: 테스트 예측 생략 (test.npy 저장 안 함)")
    ap.add_argument("--fold", type=int, default=None,
                    help="fold 하나만 실행 → results/<대회>/<모델>/folds/<k>/ 에 저장 (results.csv 는 전체 fold 를 모아 합칠 때)")


def run_argv(args: argparse.Namespace, models: list[str] | None = None) -> list[str]:
    """add_run_args 로 받은 인자를 `km run` 명령행으로 되돌림 (대회 이름 제외)."""
    out = [*(args.models if models is None else models), "--note", args.note]
    if args.max_folds:
        out += ["--max-folds", str(args.max_folds)]
    if args.no_test:
        out.append("--no-test")
    if args.fold is not None:
        out += ["--fold", str(args.fold)]
    return out


def bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in CODE_PATHS:
            tar.add(ROOT / p, arcname=p,
                    filter=lambda t: None if "__pycache__" in t.name or "/data" in t.name else t)
    return base64.b64encode(buf.getvalue()).decode()


def pip_args(models: list[str]) -> list[list[str]]:
    out = []
    for m in models:
        out += [a for a in KAGGLE_PIP.get(base_name(m), []) if a not in out]
    return out


def remote_script(args: argparse.Namespace, body: str) -> str:
    """원격 실행 코드: body 앞에 CODE(코드 묶음)·COMP·TRAIN(학습 파일 이름)·ARGS(`km run` 인자)·PIP(설치 목록) 변수를 붙임."""
    gated = [m for m in args.models if base_name(m) in GATED]
    if gated:  # Kaggle·Colab 에는 HF 토큰도 Volume 도 없어 매 fold 401 로 실패
        raise SystemExit(f"{', '.join(gated)}: HF gated 가중치가 필요한 모델이라 `km databricks` 에서만 실행 가능")
    return (f"CODE = {bundle()!r}\nCOMP = {args.comp!r}\nTRAIN = {load_competition(args.comp).train_file!r}\n"
            f"ARGS = {run_argv(args)!r}\nPIP = {pip_args(args.models)!r}\n\n" + body)
