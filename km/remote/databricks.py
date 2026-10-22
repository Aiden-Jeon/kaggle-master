"""Databricks AI Runtime(서버리스 GPU)에서 `km run` 을 실행하고 결과를 로컬 results/ 로 받아옴.

예: uv run km databricks spaceship-titanic lgbm catboost tabpfn_v2 tabicl
    uv run km databricks playground-series-s6e10 tabpfn_v2 --accelerator GPU_8xH100 --no-wait
    uv run km databricks playground-series-s6e10 --fetch <run_id>

준비: databricks CLI 로그인 + databricks.local.yaml (환경별 설정이라 git 에서 제외)
    profile: <~/.databrickscfg 프로필>
    volume: /Volumes/<catalog>/<schema>/<volume>   # 대회 데이터와 실행 결과를 둘 UC Volume

동작: 대회 데이터를 Volume 에 한 번 올려 두고 (없으면 로컬에서 kaggle 로 받아 업로드),
     코드 스냅샷과 함께 `databricks air run` 으로 제출 → 로그 스트리밍 → Volume 의 results/ 를 받아 병합.
     Kaggle 토큰은 워크스페이스로 보내지 않음.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

from km.competition import ROOT, load_competition
from km.remote.common import CODE_PATHS, add_run_args, pip_args, run_argv
from km.results import RESULTS, merge_results
from km.download import download

LOCAL_CONFIG = ROOT / "databricks.local.yaml"
RUNS = RESULTS / "_air" / "runs.json"  # run_id → 결과 위치 (나중에 --fetch 용)
BASE_PIP = [["scikit-learn", "lightgbm", "xgboost", "catboost"]]  # AI Runtime 기본 환경에는 없음

RUNNER = '''\
import shutil, subprocess, sys
from pathlib import Path

SRC = Path("/tmp/km")  # code_source 를 쓰기 가능한 곳으로 복사 (folds.csv, results/ 생성)
shutil.copytree(Path(__file__).parent, SRC)
data = SRC / "competitions" / COMP / "data"
data.mkdir(parents=True, exist_ok=True)
for f in Path(DATA).iterdir():
    (data / f.name).symlink_to(f)

out = Path(OUT)
out.mkdir(parents=True, exist_ok=True)
for args in PIP:  # 모델별 설치 실패는 해당 모델만 실패하도록 계속 진행
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args])
subprocess.run([sys.executable, "-m", "pip", "list"], stdout=open(out / "pip_list.txt", "w"))
subprocess.run(["nvidia-smi"])
code = subprocess.run([sys.executable, "-u", "-m", "km", "run", COMP, *ARGS], cwd=SRC).returncode
if (SRC / "results").exists():
    shutil.copytree(SRC / "results", out / "results", dirs_exist_ok=True)
print(f"km run 종료 코드: {code}")
sys.exit(code)
'''


def settings(args) -> dict:
    cfg = yaml.safe_load(LOCAL_CONFIG.read_text()) if LOCAL_CONFIG.exists() else {}
    cfg.update({k: v for k, v in dict(profile=args.profile, volume=args.volume).items() if v})
    missing = [k for k in ("profile", "volume") if not cfg.get(k)]
    if missing:
        raise SystemExit(f"{', '.join(missing)} 설정 없음: {LOCAL_CONFIG.name} 에 적거나 --{missing[0]} 로 지정")
    cfg["volume"] = cfg["volume"].rstrip("/")
    return cfg


def databricks(profile: str, *args, capture=False, check=True, timeout=600):
    return subprocess.run(["databricks", *args, "-p", profile], check=check, text=True,
                          capture_output=capture, timeout=timeout)


def ensure_data(comp, profile: str, volume: str) -> str:
    """Volume 에 대회 데이터가 없으면 로컬에서 받아 업로드. Volume 경로를 반환."""
    remote = f"{volume}/data/{comp.name}"
    ls = databricks(profile, "fs", "ls", f"dbfs:{remote}", capture=True, check=False)
    if ls.returncode == 0 and comp.train_file in ls.stdout.splitlines():
        return remote
    if not (comp.data_dir / comp.train_file).exists():
        download(comp)
    print(f"데이터 업로드: {comp.data_dir} → {remote}")
    databricks(profile, "fs", "cp", "-r", "--overwrite", str(comp.data_dir), f"dbfs:{remote}", timeout=3600)
    return remote


def snapshot(work: Path, comp: str, run_args: list[str], models: list[str], data: str, out: str):
    code = work / "km"
    for p in CODE_PATHS:
        src = ROOT / p
        if src.is_dir():
            shutil.copytree(src, code / p, ignore=shutil.ignore_patterns("__pycache__", "data"))
        else:
            shutil.copy(src, code / p)
    consts = dict(COMP=comp, ARGS=run_args, PIP=BASE_PIP + pip_args(models), DATA=data, OUT=out)
    (code / "air_runner.py").write_text("".join(f"{k} = {v!r}\n" for k, v in consts.items()) + "\n" + RUNNER)
    return code


def submit(cfg: dict, comp: str, run_args: list[str], models: list[str], accelerator: str, timeout: int,
           tag: str) -> str:
    n_gpus = re.match(r"GPU_(\d+)x", accelerator)
    if not n_gpus:
        raise SystemExit(f"accelerator 형식이 GPU_<n>x<type> 이 아님: {accelerator}")
    key = f"{tag + '-' if tag else ''}{comp}-{datetime.now():%Y%m%d-%H%M%S}"
    out = f"{cfg['volume']}/runs/{key}"
    data = ensure_data(load_competition(comp), cfg["profile"], cfg["volume"])

    work = Path(tempfile.mkdtemp())
    code = snapshot(work, comp, run_args, models, data, out)
    (work / "air.yaml").write_text(yaml.safe_dump({
        "experiment_name": f"km-{comp}", "mlflow_run_name": key,
        "compute": {"accelerator_type": accelerator, "num_accelerators": int(n_gpus[1])},
        "environment": {"version": 6},
        "code_source": {"type": "snapshot", "snapshot": {"root_path": str(code)}},
        "command": "python $CODE_SOURCE_PATH/air_runner.py",
        "timeout_minutes": timeout, "max_retries": 0,  # 실패 시 재시도하면 같은 실험을 반복하므로 끔
    }, sort_keys=False))
    res = databricks(cfg["profile"], "air", "run", "-f", str(work / "air.yaml"), "-o", "json", capture=True)
    shutil.rmtree(work)
    info = json.loads(res.stdout[res.stdout.index("{"):])["data"]  # 앞에 업로드 진행 메시지가 섞임

    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    runs[info["run_id"]] = dict(key=key, comp=comp, models=models, out=out, url=info["dashboard_url"])
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    RUNS.write_text(json.dumps(runs, indent=2))
    print(f"제출 완료: run_id={info['run_id']}\n  {info['dashboard_url']}")
    return info["run_id"]


def status(profile: str, run_id: str) -> str:
    out = databricks(profile, "air", "get", run_id, "-o", "json", capture=True).stdout
    return json.loads(out[out.index("{"):])["data"]["status"]


def wait(profile: str, run_id: str, poll: int = 30) -> str:
    while (s := status(profile, run_id)) in ("PENDING", "QUEUED", "BLOCKED"):
        print(f"  {time.strftime('%H:%M:%S')} {s}")
        time.sleep(poll)
    databricks(profile, "air", "logs", run_id, check=False, timeout=None)  # 끝날 때까지 로그 스트리밍
    while (s := status(profile, run_id)) in ("PENDING", "QUEUED", "BLOCKED", "RUNNING"):  # 스트림이 끊긴 경우
        time.sleep(poll)
    return s


def fetch(profile: str, run_id: str):
    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    if run_id not in runs:
        raise SystemExit(f"{RUNS} 에 run_id {run_id} 기록 없음")
    dl = Path(tempfile.mkdtemp())
    res = databricks(profile, "fs", "cp", "-r", f"dbfs:{runs[run_id]['out']}/results", str(dl / "results"),
                     capture=True, check=False, timeout=3600)
    if res.returncode != 0 or not (dl / "results" / "results.csv").exists():
        print(f"results 없음. 로그 확인: databricks air logs {run_id} -p {profile}\n{res.stderr.strip()}")
        return
    merge_results(dl / "results", "databricks")
    shutil.rmtree(dl)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km databricks")
    add_run_args(ap, models_nargs="*")  # --fetch 일 때는 models 생략
    ap.add_argument("--no-wait", action="store_true", help="제출만 하고 종료 (나중에 --fetch 로 수집)")
    ap.add_argument("--fetch", metavar="RUN_ID", help="이미 끝난 run 의 결과만 받아오기")
    ap.add_argument("--accelerator", default="GPU_1xA10", help="GPU 종류 (GPU_1xA10, GPU_8xH100 등)")
    ap.add_argument("--timeout", type=int, default=720, help="실행 시간 제한 (분)")
    ap.add_argument("--tag", default="", help="실행 이름 태그 (MLflow run 이름·결과 폴더에 붙음)")
    ap.add_argument("--profile", help=f"databricks 프로필 ({LOCAL_CONFIG.name} 값을 덮어씀)")
    ap.add_argument("--volume", help=f"UC Volume 경로 ({LOCAL_CONFIG.name} 값을 덮어씀)")
    args = ap.parse_args(argv)

    cfg = settings(args)
    load_competition(args.comp)  # 설정 존재 확인
    run_id = args.fetch
    if not run_id:
        if not args.models:
            ap.error("models 가 필요함 (--fetch 가 아닐 때)")
        run_id = submit(cfg, args.comp, run_argv(args), args.models, args.accelerator, args.timeout, args.tag)
        if args.no_wait:
            return
        print(f"최종 상태: {wait(cfg['profile'], run_id)}")
    fetch(cfg["profile"], run_id)

