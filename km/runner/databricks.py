"""Databricks AI Runtime(서버리스 GPU)에서 `km run` 을 실행하고 결과를 로컬 results/ 로 받아옴.

예: uv run km databricks spaceship-titanic lgbm catboost tabpfn_v2 tabicl
    uv run km databricks playground-series-s6e10 tabpfn_v2 --no-wait
    uv run km databricks playground-series-s6e10 --fetch <run_id>
    uv run km databricks allstate-claims-severity mitra_ft --split-folds   # fold 마다 GPU 작업 하나씩 동시에

준비: databricks CLI 로그인 + databricks.local.yaml (환경별 설정이라 git 에서 제외)
    profile: <~/.databrickscfg 프로필>
    volume: /Volumes/<catalog>/<schema>/<volume>   # 대회 데이터와 실행 결과를 둘 UC Volume
    accelerator: GPU_1xH100                         # 선택. 기본 GPU 종류 (--accelerator 가 우선)
    image: <catalog>.<schema>.km-air:v4             # 선택. 패키지를 미리 설치한 이미지 (docker/air). 없으면 실행마다 pip 설치

동작: 대회 데이터를 Volume 에 한 번 올려 두고 (없으면 로컬에서 kaggle 로 받아 업로드),
     코드 스냅샷과 함께 `databricks air run` 으로 제출 → 로그 스트리밍 → Volume 의 results/ 를 받아 병합.
     GBDT(lgbm·xgb·catboost)는 GPU 를 안 쓰므로 CPU serverless 일회성 작업(`databricks jobs submit`)으로 따로 제출.
     모델 가중치는 Volume 의 cache/models.tar 에 모아 두고 실행마다 다시 받지 않음.
     Kaggle·HF 토큰은 워크스페이스로 보내지 않음 (HF gated 가중치는 로컬에서 받아 Volume 의 weights/ 에 올림).
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

from km.core.competition import ROOT, load_competition
from km.core.download import download
from km.core.results import OUTPUTS, merge_results
from km.models import GATED, base_name
from km.runner.common import CODE_PATHS, add_run_args, check_fold_args, pip_args, run_argv

LOCAL_CONFIG = ROOT / "databricks.local.yaml"
DEFAULT_ACCELERATOR = "GPU_1xH100"  # A10G(24GB)는 TFM 이 OOM 재시도를 반복해 느림
RUNS = OUTPUTS / "_air" / "runs.json"  # run_id → 결과 위치 (나중에 --fetch 용, 어느 worktree 에서 제출했든 수집 가능)
BASE_PIP = [["scikit-learn", "lightgbm", "xgboost", "catboost"]]  # AI Runtime 기본 환경에는 없음

# GPU 를 안 쓰는 모델 → CPU serverless 작업으로 따로 실행. GPU 노드에서 TFM 과 같이 돌리면 CPU 를 나눠 쓰느라
# 극도로 느림 (2026-10-23 8xH100: Restaurant 110행 xgb fold 하나 593s) → GPU 시간만 잡아먹음
CPU_MODELS = {"lgbm", "xgb", "catboost"}
CPU_ENV = "4"  # serverless environment version (Python 3.12)
CPU_PACKAGES = {"numpy", "pandas", "scipy", "scikit-learn", "pyyaml", "lightgbm", "xgboost", "catboost"}

MODEL_CACHE = "cache/models.tar"  # Volume 안의 모델 가중치 캐시 (Hugging Face·TabPFN)

RUNNER = '''\
import contextlib, importlib.metadata, os, re, shutil, subprocess, sys, tarfile, time, uuid
from pathlib import Path

T0 = time.time()


def step(msg):
    print(f"[runner] {msg} ({time.time() - T0:.0f}s)", flush=True)


SRC = Path("/tmp/km")  # code_source 를 쓰기 가능한 곳으로 복사 (folds.csv, results/ 생성)
shutil.copytree(Path(sys.argv[0]).parent, SRC)  # serverless 작업(CPU)에는 __file__ 이 없음
data = SRC / "competitions" / COMP / "data"
data.mkdir(parents=True, exist_ok=True)
for f in Path(DATA).iterdir():
    (data / f.name).symlink_to(f)


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def dist_name(spec):  # "pkg[extra]>=1", "pkg @ git+...", "git+https://.../Repo.git@sha" → 배포 이름
    if spec.startswith("git+"):
        return spec.split("#")[0].rstrip("/").rsplit("/", 1)[-1].split("@")[0].removesuffix(".git")
    return re.split(r"[\\s\\[<>=!~@;]", spec, maxsplit=1)[0]


INSTALLED = {canon(d.metadata["Name"]) for d in importlib.metadata.distributions()}
out = Path(OUT)
out.mkdir(parents=True, exist_ok=True)
for args in PIP:  # 모델별 설치 실패는 해당 모델만 실패하도록 계속 진행
    # 이미지에서는 이미 설치된 패키지는 건너뜀 (이미지를 다시 빌드하지 않고 추가한 모델만 설치).
    # 기본 환경에서는 -U 업그레이드 등 지정한 그대로 설치
    if IMAGE and all(canon(dist_name(a)) in INSTALLED for a in args if not a.startswith("-")):
        continue
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args])
(out / "pip_list.txt").write_text("\\n".join(sorted(f"{d.metadata['Name']}=={d.version}"
                                                   for d in importlib.metadata.distributions())) + "\\n")
step("패키지 준비")

# 모델 가중치: Volume 의 tar 를 로컬 디스크에 풀어 쓰고 (XDG_CACHE_HOME 아래 huggingface/, tabpfn/),
# km run 이 성공하고 새로 받은 가중치가 있으면 tar 를 다시 만듦. 캐시는 있으면 좋은 것 → 읽기·쓰기 실패는 경고만.
# 동시에 끝난 실행끼리는 나중 것이 덮어씀 (빠진 가중치는 다음에 다시 받음)
cache = Path("/tmp/model_cache")
cache.mkdir(parents=True, exist_ok=True)
try:
    if CACHE and Path(CACHE).exists():  # CPU 작업은 가중치가 필요 없어 CACHE 없음
        with tarfile.open(CACHE) as tar:
            tar.extractall(cache, filter="fully_trusted")  # HF 캐시는 snapshots → blobs 심볼릭 링크
except (tarfile.TarError, OSError, EOFError) as e:
    print(f"[runner] 모델 캐시 읽기 실패 → 가중치를 새로 받음: {e!r}", flush=True)
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True, exist_ok=True)


def weights():  # .locks 와 덜 받은 파일(.incomplete)은 캐시에 넣지 않음
    return {p for d in ("huggingface", "tabpfn") for p in (cache / d).rglob("*")
            if p.is_file() and ".locks" not in p.parts and p.suffix != ".incomplete"}


def keep(t):
    return None if ".locks" in t.name or t.name.endswith(".incomplete") else t


before = weights()
step(f"모델 캐시 복원 ({len(before)}개 파일)")
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"])
env = {**os.environ, "XDG_CACHE_HOME": str(cache),
       "KM_WEIGHTS": WEIGHTS}  # gated 가중치 위치 (km/models/tfm.py gated_file)
code = subprocess.run([sys.executable, "-u", "-m", "km", "run", COMP, *ARGS], cwd=SRC, env=env).returncode
step("km run 종료")
if (SRC / "results").exists():
    shutil.copytree(SRC / "results", out / "results", dirs_exist_ok=True)
new = weights() - before
if code == 0 and new and CACHE:
    tmp = Path(f"{CACHE}.{uuid.uuid4().hex}.tmp")  # 컨테이너마다 PID 가 같을 수 있어 uuid
    try:
        Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp, "w") as tar:
            for d in ("huggingface", "tabpfn"):
                if (cache / d).exists():
                    tar.add(cache / d, arcname=d, filter=keep)
        os.replace(tmp, CACHE)
        step(f"모델 캐시 갱신 (새 파일 {len(new)}개)")
    except (tarfile.TarError, OSError) as e:
        print(f"[runner] 모델 캐시 갱신 실패 (결과에는 영향 없음): {e!r}", flush=True)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
print(f"km run 종료 코드: {code}")
if code:  # serverless(CPU) 작업은 IPython 위에서 돌아 sys.exit(0) 도 실패로 처리됨
    sys.exit(code)
'''


def settings(args) -> dict:
    cfg = yaml.safe_load(LOCAL_CONFIG.read_text()) if LOCAL_CONFIG.exists() else {}
    cfg.update({k: v for k, v in dict(profile=args.profile, volume=args.volume, accelerator=args.accelerator,
                                      image=args.image).items() if v})
    cfg.setdefault("accelerator", DEFAULT_ACCELERATOR)
    if str(cfg.get("image")).lower() == "none":  # --image none / YAML 의 None·null
        cfg["image"] = None
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


def ensure_weights(models: list[str], profile: str, volume: str) -> str:
    """모델이 쓰는 HF gated 가중치가 Volume 에 없으면 로컬(HF 로그인)에서 받아 업로드. weights 폴더 경로를 반환."""
    remote = f"{volume}/weights"
    for repo, filename in dict.fromkeys(f for m in models for f in GATED.get(base_name(m), [])):
        dst = f"dbfs:{remote}/{repo}/{filename}"
        ls = databricks(profile, "fs", "ls", f"dbfs:{remote}/{repo}", capture=True, check=False)
        if ls.returncode == 0 and filename in ls.stdout.splitlines():  # 파일 경로 ls 는 실패하므로 폴더를 봄
            continue
        from huggingface_hub import hf_hub_download

        print(f"가중치 업로드: {repo}/{filename} → {remote}")
        databricks(profile, "fs", "mkdir", f"dbfs:{remote}/{repo}")  # 파일 cp 는 상위 폴더를 만들지 않음
        databricks(profile, "fs", "cp", "--overwrite", hf_hub_download(repo, filename), dst, timeout=3600)
    return remote


def snapshot(work: Path, comp: str, run_args: list[str], models: list[str], data: str, out: str, cache: str,
             preinstalled: bool, weights: str):
    code = work / "km"
    for p in CODE_PATHS:
        src = ROOT / p
        if src.is_dir():
            shutil.copytree(src, code / p, ignore=shutil.ignore_patterns("__pycache__", "data"))
        else:
            shutil.copy(src, code / p)
    consts = dict(COMP=comp, ARGS=run_args, PIP=BASE_PIP + pip_args(models), IMAGE=preinstalled, DATA=data, OUT=out,
                  CACHE=cache, WEIGHTS=weights)
    (code / "air_runner.py").write_text("".join(f"{k} = {v!r}\n" for k, v in consts.items()) + "\n" + RUNNER)
    return code


def run_key(comp: str, tag: str, kind: str = "") -> str:
    """실행 이름 = MLflow run 이름·Volume 결과 폴더 이름."""
    return f"{tag + '-' if tag else ''}{comp}{'-' + kind if kind else ''}-{datetime.now():%Y%m%d-%H%M%S}"


def submit(cfg: dict, comp: str, data: str, run_args: list[str], models: list[str], accelerator: str, timeout: int,
           tag: str) -> str:
    n_gpus = re.match(r"GPU_(\d+)x", accelerator)
    if not n_gpus:
        raise SystemExit(f"accelerator 형식이 GPU_<n>x<type> 이 아님: {accelerator}")
    key = run_key(comp, tag)
    out = f"{cfg['volume']}/runs/{key}"

    # 이미지에는 패키지가 미리 설치돼 있음 (docker/air) → 러너가 이미 있는 패키지는 설치를 건너뜀
    image = cfg.get("image")
    work = Path(tempfile.mkdtemp())
    try:
        code = snapshot(work, comp, run_args, models, data, out, f"{cfg['volume']}/{MODEL_CACHE}", bool(image),
                        ensure_weights(models, cfg["profile"], cfg["volume"]))
        (work / "air.yaml").write_text(yaml.safe_dump({
            "experiment_name": f"km-{comp}", "mlflow_run_name": key,
            "compute": {"accelerator_type": accelerator, "num_accelerators": int(n_gpus[1])},
            "environment": {"unity_catalog_image": image} if image else {"version": 6},
            "code_source": {"type": "snapshot", "snapshot": {"root_path": str(code)}},
            "command": "python $CODE_SOURCE_PATH/air_runner.py",
            "timeout_minutes": timeout, "max_retries": 0,  # 실패 시 재시도하면 같은 실험을 반복하므로 끔
        }, sort_keys=False))
        res = databricks(cfg["profile"], "air", "run", "-f", str(work / "air.yaml"), "-o", "json", capture=True)
    finally:
        shutil.rmtree(work)
    info = json.loads(res.stdout[res.stdout.index("{"):])["data"]  # 앞에 업로드 진행 메시지가 섞임

    record(info["run_id"], dict(key=key, comp=comp, models=models, out=out, url=info["dashboard_url"]))
    return info["run_id"]


def record(run_id: str, run: dict):
    """runs.json 은 메인 체크아웃 하나를 모든 worktree 가 같이 씀 (OUTPUTS) → 잠금을 걸고 읽은 뒤 씀.
    임시 파일 → 교체로 써서 잠금 없이 읽는 fetch 가 쓰다 만 파일을 보지 않음."""
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS.parent / ".runs.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
        runs[run_id] = run
        tmp = RUNS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(runs, indent=2))
        os.replace(tmp, RUNS)
    print(f"제출 완료: run_id={run_id}\n  {run['url']}")


def cpu_dependencies() -> list[str]:
    """CPU 작업 환경 패키지: GPU 이미지(docker/air/requirements.txt)와 같은 줄(같은 버전). 하나라도 없으면 중단."""
    found = {}
    for line in (ROOT / "docker" / "air" / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        name = re.sub(r"[-_.]+", "-", re.split(r"[\s\[<>=!~@;]", line, maxsplit=1)[0]).lower()
        if name in CPU_PACKAGES:
            found[name] = line
    if missing := CPU_PACKAGES - found.keys():
        raise SystemExit(f"docker/air/requirements.txt 에 CPU 작업 패키지 없음: {', '.join(sorted(missing))}")
    return list(found.values())


def submit_cpu(cfg: dict, comp: str, data: str, run_args: list[str], models: list[str], timeout: int,
               tag: str) -> str:
    """CPU serverless 일회성 작업으로 제출. 같은 러너를 쓰고 패키지는 작업 환경이 설치 (러너는 설치를 건너뜀).
    코드는 Workspace 에 올림 (`spark_python_task` 는 Workspace 파일 실행) → fetch 에서 지움."""
    key = run_key(comp, tag, "cpu")
    out = f"{cfg['volume']}/runs/{key}"
    me = json.loads(databricks(cfg["profile"], "current-user", "me", "-o", "json", capture=True).stdout)["userName"]
    remote = f"/Workspace/Users/{me}/.km/{key}"
    work = Path(tempfile.mkdtemp())
    try:
        code = snapshot(work, comp, run_args, models, data, out, cache="", preinstalled=True, weights="")
        databricks(cfg["profile"], "workspace", "import-dir", str(code), remote, capture=True, timeout=1200)
        (work / "job.json").write_text(json.dumps({
            "run_name": f"km-{key}", "timeout_seconds": timeout * 60,
            "tasks": [{"task_key": "km", "environment_key": "km",
                       "spark_python_task": {"python_file": f"{remote}/air_runner.py"}}],
            "environments": [{"environment_key": "km",
                              "spec": {"environment_version": CPU_ENV, "dependencies": cpu_dependencies()}}],
        }))
        res = databricks(cfg["profile"], "jobs", "submit", "--no-wait", "--json", f"@{work / 'job.json'}", "-o",
                         "json", capture=True)
    except BaseException:
        databricks(cfg["profile"], "workspace", "delete", "--recursive", remote, capture=True, check=False)
        raise
    finally:
        shutil.rmtree(work)
    run_id = str(json.loads(res.stdout)["run_id"])
    run = cli_json(cfg["profile"], "jobs", "get-run", run_id)
    record(run_id, dict(key=key, comp=comp, models=models, out=out, url=run["run_page_url"], kind="cpu",
                        code=remote))
    return run_id


def cli_json(profile: str, *args, tries: int = 10, poll: int = 30) -> dict:
    """상태 조회 (JSON). 토큰 갱신·일시적 5xx 같은 CLI 실패는 poll 초 간격으로 tries 번까지 재시도."""
    for i in range(1, tries + 1):
        try:
            res = databricks(profile, *args, "-o", "json", capture=True, check=False, timeout=120)
            if res.returncode == 0:
                return json.loads(res.stdout[res.stdout.index("{"):])
            err = res.stderr.strip()[-200:]
        except subprocess.TimeoutExpired:
            err = "120초 응답 없음"
        if i == tries:
            raise RuntimeError(f"`databricks {' '.join(args)}` {tries}번 실패: {err}")
        print(f"  {time.strftime('%H:%M:%S')} `databricks {' '.join(args[:2])}` 실패 ({i}/{tries}), 재시도: {err}")
        time.sleep(poll)


def status(profile: str, run_id: str) -> str:
    return cli_json(profile, "air", "get", run_id)["data"]["status"]


def wait(profile: str, run_id: str, poll: int = 30) -> str:
    while (s := status(profile, run_id)) in ("PENDING", "QUEUED", "BLOCKED"):
        print(f"  {time.strftime('%H:%M:%S')} {s}")
        time.sleep(poll)
    databricks(profile, "air", "logs", run_id, check=False, timeout=None)  # 끝날 때까지 로그 스트리밍
    while (s := status(profile, run_id)) in ("PENDING", "QUEUED", "BLOCKED", "RUNNING"):  # 스트림이 끊긴 경우
        time.sleep(poll)
    return s


def wait_cpu(profile: str, run_id: str, poll: int = 30) -> str:
    """CPU 작업은 로그 스트리밍이 없음 → 상태가 바뀔 때만 출력하고, 끝나면 로그(뒷부분)를 한 번에 출력."""
    last = None
    while True:
        run = cli_json(profile, "jobs", "get-run", run_id)
        state = run["state"]
        if state["life_cycle_state"] != last:
            last = state["life_cycle_state"]
            print(f"  {time.strftime('%H:%M:%S')} CPU {last}")
        if last in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        time.sleep(poll)
    if tasks := run.get("tasks"):  # 환경 준비에서 실패하면 task 실행이 없을 수 있음
        res = databricks(profile, "jobs", "get-run-output", str(tasks[0]["run_id"]), "-o", "json", capture=True,
                         check=False)
        output = json.loads(res.stdout) if res.returncode == 0 else {}
        print("\n".join(output.get(k) or "" for k in ("logs", "error", "error_trace")).strip())
    if state.get("state_message"):
        print(f"CPU 작업 상태 메시지: {state['state_message']}")
    return state.get("result_state", last)


def fetch(profile: str, run_id: str):
    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    if run_id not in runs:
        raise SystemExit(f"{RUNS} 에 run_id {run_id} 기록 없음")
    dl = Path(tempfile.mkdtemp())
    res = databricks(profile, "fs", "cp", "-r", f"dbfs:{runs[run_id]['out']}/results", str(dl / "results"),
                     capture=True, check=False, timeout=3600)
    run = runs[run_id]
    if run.get("code"):  # CPU 작업 코드 (Workspace). 끝난 작업에만 fetch 하므로 더는 필요 없음
        databricks(profile, "workspace", "delete", "--recursive", run["code"], capture=True, check=False)
    # --fold 실행은 results.csv 없이 <대회>/<모델>/folds/<k>/ 만 옴
    folds = list((dl / "results").glob("*/*/folds/*/meta.json"))
    if res.returncode != 0 or not ((dl / "results" / "results.csv").exists() or folds):
        logs = (f"{run['url']} 또는 databricks jobs get-run {run_id} -p {profile}" if run.get("kind") == "cpu"
                else f"databricks air logs {run_id} -p {profile}")
        print(f"results 없음. 로그 확인: {logs}\n{res.stderr.strip()}")
        shutil.rmtree(dl)
        return
    merge_results(dl / "results", "databricks-cpu" if run.get("kind") == "cpu" else "databricks")
    if folds:  # 모델 하나가 실패해도 다른 모델의 fold 는 옴 → 빠진 fold 는 합칠 때 알림. 그때 볼 로그 위치
        print(f"fold 결과 {len(folds)}개 수집. 로그: databricks air logs {run_id} -p {profile}")
    shutil.rmtree(dl)


def finish(profile: str, comp: str, run_id: str, is_cpu: bool):
    """끝날 때까지 기다린 뒤 결과 수집. 실패해도 다른 작업 수집은 계속되도록 예외는 안내만 출력."""
    kind = "CPU" if is_cpu else "GPU"
    try:
        print(f"최종 상태 ({kind}): {wait_cpu(profile, run_id) if is_cpu else wait(profile, run_id)}")
        fetch(profile, run_id)
    except Exception as e:
        print(f"{kind} 작업 {run_id} 수집 실패: {e!r}\n  나중에: km databricks {comp} --fetch {run_id}")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km databricks")
    add_run_args(ap, models_nargs="*")  # --fetch 일 때는 models 생략
    ap.add_argument("--no-wait", action="store_true", help="제출만 하고 종료 (나중에 --fetch 로 수집)")
    ap.add_argument("--fetch", metavar="RUN_ID", help="이미 끝난 run 의 결과만 받아오기")
    ap.add_argument("--accelerator", help=f"GPU 종류 (GPU_1xA10, GPU_8xH100 등). 기본: {LOCAL_CONFIG.name} 의 "
                                          f"accelerator, 없으면 {DEFAULT_ACCELERATOR}")
    ap.add_argument("--image", help=f"실행 이미지 <catalog>.<schema>.<image>:<tag> ({LOCAL_CONFIG.name} 의 image 를 "
                                    "덮어씀). none 이면 기본 환경에서 pip 설치")
    ap.add_argument("--timeout", type=int, default=720, help="실행 시간 제한 (분)")
    ap.add_argument("--tag", default="", help="실행 이름 태그 (MLflow run 이름·결과 폴더에 붙음)")
    ap.add_argument("--profile", help=f"databricks 프로필 ({LOCAL_CONFIG.name} 값을 덮어씀)")
    ap.add_argument("--volume", help=f"UC Volume 경로 ({LOCAL_CONFIG.name} 값을 덮어씀)")
    ap.add_argument("--split-folds", action="store_true",
                    help="GPU 모델을 fold 마다 작업 하나로 나눠 동시에 제출 (GPU 작업이 fold 수만큼). 다 끝나면 fold 를 합쳐 "
                         "results.csv 에 기록. 벽시계 시간은 줄지만 GPU 시간은 작업마다 준비 시간만큼 늘어남")
    args = ap.parse_args(argv)
    check_fold_args(ap, args)
    if args.split_folds and (args.fold is not None or args.max_folds or args.fold_batch):
        ap.error("--split-folds 는 --fold·--max-folds·--fold-batch 와 같이 쓸 수 없음")

    cfg = settings(args)
    comp = load_competition(args.comp)  # 설정 존재 확인
    from km.runner.run import merge_ready  # km.models 까지 불러오므로 필요할 때만

    if args.fetch:
        fetch(cfg["profile"], args.fetch)
        merge_ready(comp, "databricks")  # fold 작업이면 마지막 fold 를 받을 때 합쳐짐
        return
    if not args.models:
        ap.error("models 가 필요함 (--fetch 가 아닐 때)")
    cpu = [m for m in args.models if base_name(m) in CPU_MODELS]
    gpu = [m for m in args.models if m not in cpu]
    if args.split_folds:
        if not gpu:
            ap.error("--split-folds 는 GPU 모델용 (GBDT 는 CPU 작업 하나로 충분히 빠름)")
        if int(re.match(r"GPU_(\d+)x", cfg["accelerator"])[1]) > 1:  # fold 작업 수 × 장수만큼 GPU 를 잡음
            ap.error(f"--split-folds 는 GPU 1장 작업으로 (지금 {cfg['accelerator']} × fold {comp.cv['n_splits']}개)")
        if not (comp.data_dir / comp.train_file).exists():  # 다 끝난 뒤 합칠 때 로컬 타깃이 필요 → 미리 받아 둠
            download(comp)
        # 예전 시도에서 남은 fold 가 이번 묶음과 섞이지 않게 (합칠 때 batch 로도 확인).
        # folds/ 는 모든 worktree 가 같이 씀 (OUTPUTS) → 다른 세션이 같은 대회·모델을 --split-folds 로 돌리는 중이면 그 fold 도 지워짐
        for name in gpu:
            if (stale := OUTPUTS / comp.name / name / "folds").exists():
                print(f"예전 fold 결과 삭제: {stale}")
                shutil.rmtree(stale)
    data = ensure_data(comp, cfg["profile"], cfg["volume"])
    runs = []  # (run_id, CPU 작업 여부)
    if gpu and args.split_folds:
        batch = f"{args.tag + '-' if args.tag else ''}{datetime.now():%Y%m%d-%H%M%S}"
        print(f"GPU 작업 {comp.cv['n_splits']}개 제출 (fold 마다 {cfg['accelerator']}, 묶음 {batch}) — 워크스페이스 GPU 쿼터 확인")
        for k in range(comp.cv["n_splits"]):  # 같은 초에 제출해도 실행 이름이 겹치지 않게 태그에 fold 번호
            fold_args = copy.copy(args)
            fold_args.fold, fold_args.fold_batch = k, batch
            runs.append((submit(cfg, comp.name, data, run_argv(fold_args, gpu), gpu, cfg["accelerator"], args.timeout,
                                f"{args.tag + '-' if args.tag else ''}fold{k}"), False))
    elif gpu:
        runs.append((submit(cfg, comp.name, data, run_argv(args, gpu), gpu, cfg["accelerator"], args.timeout,
                            args.tag), False))
    if cpu:
        runs.append((submit_cpu(cfg, comp.name, data, run_argv(args, cpu), cpu, args.timeout, args.tag), True))
    if args.no_wait:
        if args.split_folds:
            print(f"끝난 뒤 작업마다 km databricks {comp.name} --fetch <run_id> → 마지막 fold 를 받을 때 자동으로 합쳐짐")
        return
    # CPU 작업은 상태만 폴링 → 스레드에서 기다리다 끝나는 대로 수집. GPU 작업은 여기서 로그를 스트리밍
    threads = [threading.Thread(target=finish, args=(cfg["profile"], comp.name, rid, True), daemon=True)
               for rid, is_cpu in runs if is_cpu]
    try:
        for t in threads:
            t.start()
        for rid, is_cpu in runs:
            if not is_cpu:
                finish(cfg["profile"], comp.name, rid, False)
        for t in threads:
            t.join()
    except KeyboardInterrupt:  # 작업은 Databricks 에서 계속 돎
        print("\n중단. 작업은 계속 실행 중 → 끝난 뒤 수집:")
        for rid, _ in runs:
            print(f"  km databricks {comp.name} --fetch {rid}")
        if args.split_folds:
            print("  (fold 작업은 마지막 fold 를 받을 때 자동으로 합쳐짐)")
        raise SystemExit(130)
    if args.split_folds:
        merge_ready(comp, "databricks", gpu)

