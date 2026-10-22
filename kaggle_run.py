"""Kaggle Notebook(GPU)에서 run.py 를 실행하고 결과를 로컬 results/ 로 받아옴.

예: uv run python kaggle_run.py spaceship-titanic lgbm xgb catboost tabpfn_v2 tabicl
    uv run python kaggle_run.py spaceship-titanic tabicl --max-folds 1 --no-wait

동작: 프로젝트 코드를 tar+base64 로 스크립트 하나에 담아 private kernel 로 push
     → 완료까지 대기 → 출력(results/) 다운로드 후 로컬 results/ 에 병합.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

from core.competition import load_competition
from core.launch import add_run_args, remote_script
from core.results import RESULTS, append_rows

RUNNER = '''\
import base64, io, os, subprocess, sys, tarfile
from pathlib import Path

SRC = Path("/kaggle/working/km")
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz").extractall(SRC)

# 대회 데이터 위치는 Kaggle 이미지에 따라 다르므로 train.csv 를 찾아 연결
train = next(Path("/kaggle/input").rglob("train.csv"))
data = SRC / "competitions" / COMP / "data"
data.mkdir(parents=True, exist_ok=True)
for f in train.parent.iterdir():
    (data / f.name).symlink_to(f)

for args in PIP:  # 모델별 설치 실패는 해당 모델만 실패하도록 계속 진행
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args])
subprocess.run([sys.executable, "-m", "pip", "list"], stdout=open("/kaggle/working/pip_list.txt", "w"))
subprocess.run(["nvidia-smi"])
subprocess.run([sys.executable, "run.py", COMP, *ARGS], cwd=SRC)
if (SRC / "results").exists():
    subprocess.run(["cp", "-r", str(SRC / "results"), "/kaggle/working/results"], check=True)
subprocess.run(["rm", "-rf", str(SRC)])
'''


def kaggle(*args, capture=False, timeout=600):
    # 네트워크가 멈추거나 맥이 잠자기에서 깨어날 때 무한 대기하지 않도록 시간 제한
    return subprocess.run(["kaggle", *args], check=True, text=True, capture_output=capture, timeout=timeout)


def username() -> str:
    out = kaggle("config", "view", capture=True).stdout
    for line in out.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"Kaggle username 을 찾지 못함:\n{out}")


def kernel_id(comp: str, user: str, tag: str = "") -> str:
    return f"{user}/km-{tag + '-' if tag else ''}{comp}"[: len(user) + 51]  # slug 는 50자 제한


def push(kernel: str, comp: str, script: str, accelerator: str | None = None):
    work = Path(tempfile.mkdtemp())
    (work / "run_kernel.py").write_text(script)
    (work / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel, "title": kernel.split("/")[1], "code_file": "run_kernel.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "competition_sources": [comp], "dataset_sources": [], "kernel_sources": [],
        **({"machine_shape": accelerator} if accelerator else {}),  # 예: NvidiaTeslaP100, NvidiaTeslaT4
    }))
    out = kaggle("kernels", "push", "-p", str(work), capture=True).stdout
    shutil.rmtree(work)
    print(out.strip())
    if "error" in out.lower():  # push 실패해도 exit code 0 을 반환함
        raise SystemExit("push 실패")


def wait(kernel: str, poll: int = 30) -> str:
    while True:
        out = kaggle("kernels", "status", kernel, capture=True).stdout.strip()
        print(f"  {time.strftime('%H:%M:%S')} {out}")
        status = out.lower()
        if any(s in status for s in ("complete", "error", "cancel")):
            return status
        time.sleep(poll)


def fetch(kernel: str):
    """kernel 출력 다운로드 → results/ 에 병합 (results.csv 는 행 추가)."""
    dl = Path(tempfile.mkdtemp())
    kaggle("kernels", "output", kernel, "-p", str(dl))
    log = next(dl.glob("*.log"), None)
    remote = dl / "results"
    if not remote.exists():
        print(f"results 없음. 로그 확인: {log}")
        return
    for d in remote.iterdir():
        if d.is_dir():
            shutil.copytree(d, RESULTS / d.name, dirs_exist_ok=True)
    rows = pd.read_csv(remote / "results.csv").assign(where="kaggle")
    append_rows(rows)
    print(rows.to_string(index=False))
    print(f"\n로그: {log}")


def main():
    ap = argparse.ArgumentParser()
    add_run_args(ap)
    ap.add_argument("--no-wait", action="store_true", help="push 만 하고 종료 (나중에 --fetch 로 수집)")
    ap.add_argument("--fetch", action="store_true", help="이미 끝난 kernel 의 결과만 받아오기")
    ap.add_argument("--accelerator", help="GPU 종류 (NvidiaTeslaT4: 2장 / NvidiaTeslaP100: 1장), 기본은 Kaggle 기본값")
    ap.add_argument("--tag", default="", help="kernel 이름 태그 (km-<tag>-<comp>). 같은 대회를 동시에 돌릴 때 충돌 방지")
    args = ap.parse_args()

    load_competition(args.comp)  # 설정 존재 확인
    kernel = kernel_id(args.comp, username(), args.tag)
    if not args.fetch:
        push(kernel, args.comp, remote_script(args, RUNNER), args.accelerator)
        print(f"push 완료: https://www.kaggle.com/code/{kernel}")
        if args.no_wait:
            return
        time.sleep(10)
        print(f"최종 상태: {wait(kernel)}")
    fetch(kernel)


if __name__ == "__main__":
    main()
