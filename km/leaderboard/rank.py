"""Kaggle 제출 기록을 모아 리더보드 순위를 계산 → results/leaderboard.csv

예: uv run km lb                 # competitions/ 아래 대회 전부
    uv run km lb spaceship-titanic titanic
    uv run km lb bioresponse --submit 2   # 아직 제출 안 한 모델을 CV 좋은 순으로 2개 제출 후 집계

순위: 리더보드에서 우리 팀을 뺀 뒤 (점수가 더 좋은 팀 수 + 1). 기준 열 `basis`:
     - public: 진행 중 대회. 리더보드의 public 점수끼리 비교 (private 은 아직 안 보임).
     - private: 종료된 대회. 리더보드 조회 API 의 기본 응답은 private 이 공개되어 있으면 private 이라, 우리 private 점수와 남의 private 점수를 비교.
       `kaggle competitions leaderboard -d` 다운로드는 종료 뒤에도 public 이라 (파일명 publicleaderboard, 2026-10-27 확인) 여기 쓰지 않음.
       CLI 에는 조회 옵션이 없어 kagglesdk 로 직접 받음 (`override_public` 을 안 줘서 기본 = private 우선).
     public·private 점수는 둘 다 기록.
모델 이름: 제출 설명에서 `lgbm`, `kumo+catboost`, `exaone@knn1000` 처럼 모델 이름으로 해석되는 첫 토큰.
         RETIRED 모델(limix2)이 들어간 제출은 순위 집계·자동 제출에서 뺌.
실행: 제출을 만든 실행의 CV 와 시각(run 열 = results.csv timestamp)을 같이 기록 → km board 가 실행 단위로 LB 를 붙임.
     설명의 `run <시각>` (km submit 기본) → `<n>-fold CV <점수>` → 제출 시각 전 마지막 실행 순으로 찾음.
     못 찾으면 이전 leaderboard.csv 에서 찾은 값을 제출 ref 로 유지. --submit 도 이미 제출한 '실행' 은 다시 내지 않음.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from km.core.competition import COMP_DIR, load_competition
from km.core.metrics import METRICS
from km.core.results import RESULTS, made_test, test_info
from km.leaderboard.submit import make_submission, parse_message, push
from km.models import REGISTRY, RETIRED, base_name


def kaggle(*args) -> str:
    return subprocess.run(["kaggle", *args], check=True, text=True, capture_output=True, timeout=600).stdout


def me() -> str:
    for line in kaggle("config", "view").splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("Kaggle username 을 찾지 못함")


def submissions(comp: str) -> list[dict]:
    """제출 기록. 읽지 못하면 빈 목록 대신 예외 → `--submit` 이 이미 낸 모델을 다시 내지 않도록.

    (2026-10-23: JSON 앞에 버전 경고 줄이 붙어 기록이 늘 빈 목록 → 이미 낸 모델 21개를 다시 제출)
    """
    text = kaggle("competitions", "submissions", "-c", comp, "--format", "json", "--page-size", "200")
    if "No submissions found" in text:
        return []
    for m in re.finditer(r"^\[", text, re.M):  # 경고 줄은 건너뛰고 '[' 로 시작하는 줄부터 JSON 배열을 찾음
        try:
            out = json.JSONDecoder().raw_decode(text, m.start())[0]
        except json.JSONDecodeError:
            continue
        if isinstance(out, list):
            break
    else:
        raise RuntimeError(f"제출 기록을 읽지 못함 (kaggle 출력: {text.strip()[-300:]!r})")
    if len(out) == 200:
        print(f"{comp}: 제출이 200개 이상 → 최근 200개만 반영")
    return out


def public_leaderboard(comp: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as d:
        kaggle("competitions", "leaderboard", "-c", comp, "-d", "-p", d, "-q")
        with zipfile.ZipFile(next(Path(d).glob("*.zip"))) as z:
            with z.open(next(n for n in z.namelist() if n.endswith(".csv"))) as f:
                return pd.read_csv(f, encoding="utf-8-sig")


def private_leaderboard(comp: str) -> pd.DataFrame:
    """리더보드 조회 API 의 기본 응답 전 페이지 (TeamId, Score). private 이 공개됐으면 private, 아니면 public.
    CLI 에는 이 응답을 받는 옵션이 없어 kagglesdk 를 직접 호출 (kaggle 패키지에 이미 들어 있음)."""
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

    api = KaggleApi()
    api.authenticate()
    rows, token = [], None
    with api.build_kaggle_client() as kc:
        while True:
            req = ApiGetLeaderboardRequest()
            req.competition_name = comp
            req.page_size = 200
            if token:
                req.page_token = token
            for attempt in range(5):  # 연속 조회 중 429 (요청 제한) 이 자주 남 → 잠깐 쉬고 다시
                try:
                    resp = kc.competitions.competition_api_client.get_leaderboard(req)
                    break
                except requests.exceptions.HTTPError as e:
                    if e.response is None or e.response.status_code != 429:
                        raise RuntimeError(f"리더보드 조회 실패: {e}") from e  # main 이 이 대회만 건너뜀
                    if attempt == 4:
                        raise RuntimeError(f"리더보드 조회 429 가 계속됨: {e}") from e
                    time.sleep(10 * (attempt + 1))
            rows += [(s.team_id, float(s.score)) for s in resp.submissions or [] if s.score not in (None, "")]
            token = resp.next_page_token
            if not token:
                break
    return pd.DataFrame(rows, columns=["TeamId", "Score"])


def model_of(sub: dict) -> str:
    for tok in re.findall(r"[\w.@+-]+", sub["description"] or ""):
        if all(base_name(p) in REGISTRY or base_name(p) in RETIRED for p in tok.split("+")):
            return tok
    return Path(sub["fileName"]).stem  # 설명에 모델 이름이 없으면 파일 이름 ('+', '@' 는 Kaggle 이 지움)


def is_retired(model: str) -> bool:
    """RETIRED 모델이 들어간 모델·앙상블 (예: limix2+kumo) → 리더보드 집계와 자동 제출에서 뺌."""
    return any(base_name(p) in RETIRED for p in model.split("+"))


def read_results() -> pd.DataFrame:
    """results.csv 전체. km lb 는 한 번만 읽고 대회마다 넘김 (prediction_runs 참고)."""
    csv = RESULTS / "results.csv"
    if not csv.exists():
        return pd.DataFrame(columns=["timestamp", "comp", "model", "cv", "n_folds", "note", "test"])
    return pd.read_csv(csv)


def prediction_runs(comp, results: pd.DataFrame) -> pd.DataFrame:
    """제출과 연결할 후보: 전체 fold 실행 중 테스트 예측을 만든 것 (탐색 실험·--no-test 재실행 제외), 시각 순.
    results: read_results() 의 results.csv 전체."""
    runs = results[(results.comp == comp.name) & (results.n_folds == comp.cv["n_splits"])].sort_values("timestamp")
    return runs[made_test(runs)]


def parse_utc(text) -> datetime:
    """시각 문자열 → UTC 기준 naive datetime. 문자열 비교 대신 이걸로 비교.
    타임존이 없으면 UTC 로 봄: results.csv timestamp·test.json run 은 datetime.now(timezone.utc) 로 쓰고,
    Kaggle 제출 날짜는 kagglesdk 가 UTC 'Z' 를 떼고 naive 로 준 값 (2026-10-27 확인). 타임존이 붙어 있으면 UTC 로 환산.
    못 읽으면 RuntimeError (조용히 틀리지 않도록 그 대회·모델 처리를 멈춤)."""
    try:
        t = datetime.fromisoformat(str(text).strip())
    except ValueError as e:
        raise RuntimeError(f"시각을 읽지 못함: {text!r}") from e
    return t.astimezone(timezone.utc).replace(tzinfo=None) if t.tzinfo else t


def submit_pending(comp_name: str, n: int, known: dict | None, subs: list[dict], results: pd.DataFrame) -> int:
    """test.npy 가 있지만 그 실행을 아직 제출 안 한 단일 모델을 CV 좋은 순으로 최대 n 개 제출. 제출한 개수 반환.
    known: 제출 ref → (cv, run), 이전 leaderboard.csv 에서 찾은 값. subs: 이 대회의 제출 목록 (submissions 결과, 다시 읽지 않음)."""
    comp = load_competition(comp_name)
    # 이미 제출한 실행: (모델, run). 실행을 알 수 없는 제출은 그 제출 시각까지의 예측을 이미 낸 것으로 봄
    # (그 뒤에 만든 예측은 낸 적이 없으니 제출 대상)
    runs = prediction_runs(comp, results)
    done, unknown_until = set(), {}
    for s in subs:
        model = model_of(s)
        run = submission_run(comp, s, runs[runs.model == model])[1] or (known or {}).get(str(s["ref"]), (None, None))[1]
        if run:
            done.add((model, parse_utc(run)))
        else:
            sent_at = parse_utc(s["date"])
            unknown_until[model] = max(unknown_until.get(model, sent_at), sent_at)
    todo = []
    for d in (RESULTS / comp_name).glob("*/"):
        t = test_info(comp_name, d.name)  # 제출할 test.npy 를 만든 실행. 출처를 모르면 자동 제출 안 함
        if not t or t["n_folds"] != comp.cv["n_splits"] or is_retired(d.name):
            continue
        try:
            run_at = parse_utc(t["run"])
        except RuntimeError as e:
            print(f"{comp_name}: {d.name} 의 test.json 을 읽지 못해 자동 제출 안 함 ({e})")
            continue
        if (d.name, run_at) not in done and run_at > unknown_until.get(d.name, datetime.min):
            todo.append((t["cv"], d.name))
    todo.sort(reverse=METRICS[comp.metric][1])
    sent = 0
    for cv, model in todo[:n]:
        print(f"{comp_name}: {model} 제출 (CV {comp.metric}={cv:.5f})")
        try:
            push(comp_name, make_submission(comp_name, [model]), model)
        except subprocess.CalledProcessError:  # 대개 하루 제출 한도 초과
            print(f"{comp_name}: 제출 실패 → 중단")
            break
        sent += 1
    if len(todo) > sent:
        print(f"{comp_name}: 남은 미제출 {len(todo) - sent}개 ({', '.join(m for _, m in todo[sent:])})")
    return sent


def wait_scored(comp_name: str, timeout: int = 900) -> list[dict]:
    """방금 낸 제출이 채점될 때까지 대기. 마지막으로 읽은 제출 목록을 돌려줌 (rank_rows 가 다시 읽지 않도록)."""
    subs = submissions(comp_name)
    for _ in range(timeout // 15):
        if not any("PENDING" in s["status"] for s in subs):
            break
        time.sleep(15)
        subs = submissions(comp_name)
    return subs


def submission_run(comp, sub: dict, runs: pd.DataFrame) -> tuple[float | None, str | None]:
    """제출을 만든 실행의 (CV, 실행 시각). runs = 그 모델의 예측 실행 (prediction_runs, 시각 순, UTC).
    1) 설명의 `run <시각>` (km submit 기본, 2026-10-27~)  2) 설명의 `<n>-fold CV <점수>` 와 같은 점수의 실행
    3) 제출 시각(UTC) 전 마지막 실행. 실행을 못 찾으면 실행 시각은 None (km board 는 모델 이름으로만 연결).
    시각 비교는 parse_utc 로 datetime 끼리 함 (문자열 비교는 타임존·형식이 다르면 틀림). 반환하는 실행 시각은 results.csv 의 문자열 그대로.
    모델 이름만으로 찾으면 같은 이름으로 다시 돌린 뒤 예전 제출에도 새 CV 가 붙음 (2026-10-26 Allstate kumo@full 평균 → 중앙값)."""
    info = parse_message(sub["description"])
    cv_ok = info.cv is not None and info.n_folds == comp.cv["n_splits"]
    if info.cv is not None and not cv_ok:  # 1-fold 시험 등 → 전체 fold 실행이 아님, 날짜로 추측하지 않음
        return None, None
    at = runs.timestamp.map(parse_utc)
    if info.run:  # 적힌 실행이 없으면 (지워짐 등) 추측하지 않음
        hit = runs[at == parse_utc(info.run)]
        return (float(hit.cv.iloc[-1]), hit.timestamp.iloc[-1]) if len(hit) else (float(info.cv) if cv_ok else None, None)
    before = runs[at <= parse_utc(sub["date"])]
    if cv_ok:  # 설명을 쓸 때와 같은 형식으로 비교 (반올림 차이 없음), 제출 뒤 실행은 제외
        hit = before[[f"{x:.5f}" == info.cv for x in before.cv]]
        return (float(hit.cv.iloc[-1]), hit.timestamp.iloc[-1]) if len(hit) else (float(info.cv), None)
    return (float(before.cv.iloc[-1]), before.timestamp.iloc[-1]) if len(before) else (None, None)


def rank_rows(comp_name: str, user: str, known: dict | None, subs: list[dict], results: pd.DataFrame) -> list[dict]:
    """known: 제출 ref → (cv, run), 이전 leaderboard.csv 에서 찾은 값. 이번에 실행을 못 찾으면 그 값을 유지.
    subs: 이 대회의 제출 목록 (submissions 결과), results: results.csv 전체 (read_results)."""
    comp = load_competition(comp_name)
    higher = METRICS[comp.metric][1]
    subs = [s for s in subs if s["publicScore"] not in (None, "")]
    if not subs:
        return []
    lb = public_leaderboard(comp_name)
    mine = lb.TeamMemberUserNames.fillna("").str.split(",").apply(lambda u: user in [x.strip() for x in u])
    priv = private_leaderboard(comp_name)
    if priv.empty:
        raise RuntimeError("리더보드 조회 결과가 비어 있음")
    # 조회 응답이 공개 리더보드와 다르면 (= private 공개됨, 종료 대회) private 기준. 같으면 public 기준
    basis = "private" if sorted(priv.Score) != sorted(lb.Score) else "public"
    if basis == "private":
        others = priv.loc[~priv.TeamId.isin(set(lb.loc[mine, "TeamId"])), "Score"].to_numpy()
        subs = [s for s in subs if s["privateScore"] not in (None, "")]
    else:
        others = lb.loc[~mine, "Score"].to_numpy()
    runs = prediction_runs(comp, results)
    rows = []
    for s in subs:
        score = float(s["privateScore"] if basis == "private" else s["publicScore"])
        rank = int(((others > score) if higher else (others < score)).sum()) + 1
        model = model_of(s)
        if is_retired(model):
            continue
        cv, run = submission_run(comp, s, runs[runs.model == model])
        if run is None and str(s["ref"]) in (known or {}):
            cv, run = known[str(s["ref"])]
        rows.append(dict(comp=comp_name, model=model, metric=comp.metric, basis=basis, cv=cv,
                         public=float(s["publicScore"]),
                         private=float(s["privateScore"]) if s["privateScore"] not in (None, "") else None,
                         rank=rank, teams=len(others) + 1, top_pct=round(100 * rank / (len(others) + 1), 1),
                         date=s["date"][:10], description=s["description"], ref=s["ref"], run=run))
    return rows


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km lb")
    ap.add_argument("comps", nargs="*", help="기본: competitions/ 아래 전부 (_ 로 시작하는 것 제외)")
    ap.add_argument("--submit", type=int, default=0, metavar="N",
                    help="대회마다 미제출 모델을 CV 순으로 N 개 제출 (대회별 하루 제출 한도 주의)")
    args = ap.parse_args(argv)
    all_comps = sorted(p.name for p in COMP_DIR.iterdir() if (p / "config.yaml").exists() and not p.name.startswith("_"))
    comps = args.comps or all_comps
    user = me()
    path = RESULTS / "leaderboard.csv"
    prev = pd.read_csv(path, dtype={"ref": str}) if path.exists() else pd.DataFrame()
    known = {r.ref: (r.cv, r.run) for r in prev.itertuples() if "run" in prev and pd.notna(r.run)}  # 이전에 찾은 실행
    results = read_results()  # results.csv 는 한 번만 읽고 대회마다 넘김
    rows, tallied = [], set()
    for c in comps:
        try:
            subs = submissions(c)  # 대회마다 한 번. 새로 제출했을 때만 wait_scored 가 다시 읽은 목록을 씀
            if args.submit and submit_pending(c, args.submit, known, subs, results):
                subs = wait_scored(c)
            rows += rank_rows(c, user, known, subs, results)
            tallied.add(c)
        except subprocess.CalledProcessError as e:  # Rules 미동의 대회 등은 건너뜀
            print(f"{c}: 건너뜀 ({e.stderr.strip().splitlines()[-1] if e.stderr else e})")
        except RuntimeError as e:  # 제출 기록을 못 읽음 → 이 대회는 제출·집계 모두 건너뜀
            print(f"{c}: 건너뜀 ({e})")
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["comp", "rank", "date"])
    if len(prev):  # 이번에 집계하지 않은 대회(지정 안 함·Kaggle 오류로 건너뜀)는 이전 행 유지, 없어진 대회는 버림
        old = prev[prev.comp.isin(all_comps) & ~prev.comp.isin(tallied) & ~prev.model.astype(str).map(is_retired)]
        df = pd.concat([old, df]).sort_values(["comp", "rank", "date"])
    df.to_csv(path, index=False)
    print(df.drop(columns=["description", "ref"]).to_string(index=False))
    print(f"\n저장: {path}")

