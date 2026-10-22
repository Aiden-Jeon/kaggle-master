"""run.py 실험을 Colab(무료 T4)에서 돌릴 노트북 생성 + 출력에서 결과 수집.

예: uv run python colab_run.py make spaceship-titanic tabpfn_v2@knn1000 tabicl@full --max-folds 1 --no-test -o run.ipynb
    uv run python colab_run.py collect output.txt   # 노트북 출력 텍스트에서 결과 행을 results/results.csv 에 추가

준비: Colab 왼쪽 열쇠(Secrets)에 KAGGLE_API_TOKEN (~/.kaggle/access_token 값) 등록 + 노트북 접근 허용.
노트북은 GPU 런타임으로 지정돼 있고, 셀 하나가 코드 압축 해제 → 대회 데이터 다운로드 → pip 설치 → run.py 실행.
결과 표는 RESULTS_BEGIN / RESULTS_END 사이에 CSV 로 출력됨 (OOF 등 큰 파일은 Colab 세션에만 남음).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd

from core.competition import load_competition
from core.launch import add_run_args, remote_script
from core.results import append_rows

BEGIN, END = "RESULTS_BEGIN", "RESULTS_END"

CELL = '''\
import base64, io, os, subprocess, sys, tarfile, zipfile
from pathlib import Path
from google.colab import userdata

SRC = Path("/content/km")
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz").extractall(SRC)

os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
for args in [["-U", "kaggle"], *PIP]:  # 모델별 설치 실패는 해당 모델만 실패하도록 계속 진행
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args])
data = SRC / "competitions" / COMP / "data"
if not (data / TRAIN).exists():
    subprocess.run(["kaggle", "competitions", "download", "-c", COMP, "-p", str(data)], check=True)
    for z in data.glob("*.zip"):
        zipfile.ZipFile(z).extractall(data)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])

p = subprocess.Popen([sys.executable, "-u", "run.py", COMP, *ARGS], cwd=SRC, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True)
for line in p.stdout:  # 진행 상황을 셀 출력에 바로 표시
    print(line, end="")
print(f"run.py 종료 코드: {p.wait()}")  # 음수 = 시그널로 강제 종료 (-9 는 대개 RAM OOM)
csv = SRC / "results" / "results.csv"
print("BEGIN")
print(csv.read_text() if csv.exists() else "")
print("END")
'''.replace('"BEGIN"', repr(BEGIN)).replace('"END"', repr(END))


def make(args: argparse.Namespace):
    load_competition(args.comp)  # 설정 존재 확인
    src = remote_script(args, CELL)
    nb = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                   "source": src.splitlines(keepends=True)}],
    }
    args.out.write_text(json.dumps(nb))
    print(f"노트북: {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


def collect(text: str):
    """출력 텍스트의 RESULTS_BEGIN~END 구간 CSV 를 results/results.csv 에 추가 (where=colab)."""
    if BEGIN not in text or END not in text:
        raise SystemExit("출력에 결과 구간이 없음 (실행이 끝나지 않았거나 실패)")
    body = text.rsplit(BEGIN, 1)[1].split(END, 1)[0].strip()
    rows = pd.read_csv(io.StringIO(body)).assign(where="colab")
    append_rows(rows)
    print(rows.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    add_run_args(m)
    m.add_argument("-o", "--out", type=Path, required=True)
    c = sub.add_parser("collect")
    c.add_argument("output", type=Path, help="Colab 셀 출력을 저장한 텍스트 파일")
    args = ap.parse_args()

    if args.cmd == "collect":
        return collect(args.output.read_text())
    make(args)


if __name__ == "__main__":
    main()
