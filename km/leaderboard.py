"""Kaggle 제출 기록을 모아 리더보드 순위를 계산 → results/leaderboard.csv

예: uv run km lb                 # competitions/ 아래 대회 전부
    uv run km lb spaceship-titanic titanic
    uv run km lb bioresponse --submit 2   # 아직 제출 안 한 모델을 CV 좋은 순으로 2개 제출 후 집계

순위: 다운로드한 public 리더보드에서 우리 팀을 뺀 뒤 (점수가 더 좋은 팀 수 + 1).
     Kaggle API 는 private 리더보드를 주지 않으므로 종료된 대회도 public 점수 기준이고, private 점수는 따로 기록.
모델 이름: 제출 설명에서 `lgbm`, `limix2+kumo`, `exaone@knn1000` 처럼 모델 이름으로 해석되는 첫 토큰.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd

from km.competition import COMP_DIR, load_competition
from km.metrics import METRICS
from km.models import REGISTRY, base_name
from km.results import RESULTS
from km.submit import make_submission, push


def kaggle(*args) -> str:
    return subprocess.run(["kaggle", *args], check=True, text=True, capture_output=True, timeout=600).stdout


def me() -> str:
    for line in kaggle("config", "view").splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("Kaggle username 을 찾지 못함")


def submissions(comp: str) -> list[dict]:
    out = json.loads(kaggle("competitions", "submissions", "-c", comp, "--format", "json", "--page-size", "200")
                     or "[]")
    if len(out) == 200:
        print(f"{comp}: 제출이 200개 이상 → 최근 200개만 반영")
    return out


def public_leaderboard(comp: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as d:
        kaggle("competitions", "leaderboard", "-c", comp, "-d", "-p", d, "-q")
        with zipfile.ZipFile(next(Path(d).glob("*.zip"))) as z:
            with z.open(next(n for n in z.namelist() if n.endswith(".csv"))) as f:
                return pd.read_csv(f, encoding="utf-8-sig")


def model_of(sub: dict) -> str:
    for tok in re.findall(r"[\w.@+-]+", sub["description"] or ""):
        if all(base_name(p) in REGISTRY for p in tok.split("+")):
            return tok
    return Path(sub["fileName"]).stem  # 설명에 모델 이름이 없으면 파일 이름 ('+', '@' 는 Kaggle 이 지움)


def latest_cv(comp, model: str) -> float | None:
    """results.csv 에서 해당 모델의 전체 fold 실행 중 가장 최근 CV."""
    csv = RESULTS / "results.csv"
    if not csv.exists():
        return None
    r = pd.read_csv(csv)
    r = r[(r.comp == comp.name) & (r.model == model) & (r.n_folds == comp.cv["n_splits"])]
    return float(r.sort_values("timestamp").cv.iloc[-1]) if len(r) else None


def submit_pending(comp_name: str, n: int) -> int:
    """test.npy 가 있지만 아직 제출 안 한 단일 모델을 CV 좋은 순으로 최대 n 개 제출. 제출한 개수 반환."""
    comp = load_competition(comp_name)
    done = {model_of(s) for s in submissions(comp_name)}
    todo = [(latest_cv(comp, d.name), d.name) for d in (RESULTS / comp_name).glob("*/")
            if (d / "test.npy").exists() and d.name not in done and latest_cv(comp, d.name) is not None]
    todo.sort(reverse=METRICS[comp.metric][1])
    sent = 0
    for cv, model in todo[:n]:
        print(f"{comp_name}: {model} 제출 (CV {comp.metric}={cv:.5f})")
        try:
            push(comp_name, make_submission(comp_name, [model]), model, f"{comp.cv['n_splits']}-fold CV {cv:.5f}")
        except subprocess.CalledProcessError:  # 대개 하루 제출 한도 초과
            print(f"{comp_name}: 제출 실패 → 중단")
            break
        sent += 1
    if len(todo) > sent:
        print(f"{comp_name}: 남은 미제출 {len(todo) - sent}개 ({', '.join(m for _, m in todo[sent:])})")
    return sent


def wait_scored(comp_name: str, timeout: int = 900):
    """방금 낸 제출이 채점될 때까지 대기."""
    for _ in range(timeout // 15):
        if not any("PENDING" in s["status"] for s in submissions(comp_name)):
            return
        time.sleep(15)


def rank_rows(comp_name: str, user: str) -> list[dict]:
    comp = load_competition(comp_name)
    higher = METRICS[comp.metric][1]
    subs = [s for s in submissions(comp_name) if s["publicScore"] not in (None, "")]
    if not subs:
        return []
    lb = public_leaderboard(comp_name)
    mine = lb.TeamMemberUserNames.fillna("").str.split(",").apply(lambda u: user in [x.strip() for x in u])
    others = lb.loc[~mine, "Score"].to_numpy()
    rows = []
    for s in subs:
        pub = float(s["publicScore"])
        rank = int(((others > pub) if higher else (others < pub)).sum()) + 1
        model = model_of(s)
        rows.append(dict(comp=comp_name, model=model, metric=comp.metric, cv=latest_cv(comp, model), public=pub,
                         private=float(s["privateScore"]) if s["privateScore"] not in (None, "") else None,
                         rank=rank, teams=len(others) + 1, top_pct=round(100 * rank / (len(others) + 1), 1),
                         date=s["date"][:10], description=s["description"], ref=s["ref"]))
    return rows


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km lb")
    ap.add_argument("comps", nargs="*", help="기본: competitions/ 아래 전부 (_ 로 시작하는 것 제외)")
    ap.add_argument("--submit", type=int, default=0, metavar="N",
                    help="대회마다 미제출 모델을 CV 순으로 N 개 제출 (대회별 하루 제출 한도 주의)")
    args = ap.parse_args(argv)
    comps = args.comps or sorted(p.name for p in COMP_DIR.iterdir()
                                 if (p / "config.yaml").exists() and not p.name.startswith("_"))
    user = me()
    rows = []
    for c in comps:
        try:
            if args.submit and submit_pending(c, args.submit):
                wait_scored(c)
            rows += rank_rows(c, user)
        except subprocess.CalledProcessError as e:  # Rules 미동의 대회 등은 건너뜀
            print(f"{c}: 건너뜀 ({e.stderr.strip().splitlines()[-1] if e.stderr else e})")
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["comp", "rank", "date"])
    path = RESULTS / "leaderboard.csv"
    if args.comps and path.exists():  # 일부 대회만 갱신할 때는 나머지 대회 행 유지
        old = pd.read_csv(path)
        df = pd.concat([old[~old.comp.isin(comps)], df]).sort_values(["comp", "rank", "date"])
    df.to_csv(path, index=False)
    print(df.drop(columns=["description", "ref"]).to_string(index=False))
    print(f"\n저장: {path}")

