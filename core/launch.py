"""run.py 를 다른 곳(GPU 별 작업자 프로세스, Kaggle Notebook, Colab)에서 실행할 때 공유하는 인자와 코드 묶음."""

from __future__ import annotations

import argparse
import base64
import io
import tarfile

from core.competition import ROOT
from models import KAGGLE_PIP, base_name

CODE_PATHS = ["core", "models", "run.py", "competitions"]


def add_run_args(ap: argparse.ArgumentParser):
    """run.py 로 그대로 전달되는 인자. 새 인자를 추가하면 run_argv 에도 반영."""
    ap.add_argument("comp")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--max-folds", type=int, default=None, help="빠른 확인용: 앞쪽 n개 fold만 실행")
    ap.add_argument("--note", default="", help="results.csv 에 남길 메모")
    ap.add_argument("--no-test", action="store_true", help="실험용: 테스트 예측 생략 (test.npy 저장 안 함)")


def run_argv(args: argparse.Namespace, models: list[str] | None = None) -> list[str]:
    """add_run_args 로 받은 인자를 run.py 명령행으로 되돌림 (대회 이름 제외)."""
    out = [*(args.models if models is None else models), "--note", args.note]
    if args.max_folds:
        out += ["--max-folds", str(args.max_folds)]
    if args.no_test:
        out.append("--no-test")
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
    """원격 실행 코드: body 앞에 CODE(코드 묶음)·COMP·ARGS(run.py 인자)·PIP(설치 목록) 변수를 붙임."""
    return (f"CODE = {bundle()!r}\nCOMP = {args.comp!r}\nARGS = {run_argv(args)!r}\nPIP = {pip_args(args.models)!r}\n\n"
            + body)
