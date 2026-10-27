"""Kaggle 제출 기록을 모아 리더보드 순위를 계산 → results/leaderboard.csv

예: uv run km lb                 # competitions/ 아래 대회 전부
    uv run km lb spaceship-titanic titanic
    uv run km lb bioresponse --submit 2   # 아직 제출 안 한 모델을 CV 좋은 순으로 2개 제출 후 집계

순위: 리더보드에서 우리 팀을 뺀 뒤 (점수가 더 좋은 팀 수 + 1). 기준 열 `basis` 는 대회 마감일로 정함:
     - public: 진행 중 대회 (마감 전. Titanic·House Prices 같은 상시 대회는 마감이 2030 년). 리더보드의 public 점수끼리 비교.
       리더보드 다운로드 (= `kaggle competitions leaderboard -d`, zip 하나) 의 TeamMemberUserNames 로 우리 팀을 찾음.
     - private: 종료된 대회. 리더보드 조회 API 의 기본 응답은 private 이 공개되어 있으면 private 이라, 우리 private 점수와 남의 private 점수를 비교.
       다운로드는 종료 뒤에도 public 이라 (파일명 publicleaderboard, 2026-10-27 확인) 받지 않음.
       `override_public` 을 안 줘서 기본 = private 우선. 마감 뒤 제출은 리더보드에 안 올라가니, 마감 전 제출이 있을 때만
       우리 팀을 찾아 뺌 (조회 API 에는 멤버가 없어 제출 기록의 팀 이름으로).
       종료됐어도 우리 제출에 private 점수가 없으면 (private 비공개) public 으로 비교.
     public·private 점수는 둘 다 기록.
Kaggle 호출: 제출 기록·리더보드·마감일·username 은 kagglesdk (sdk + sdk_call: 429 재시도, 오류는 RuntimeError → 그 대회만 건너뜀).
     제출 (push) 만 kaggle CLI.
모델 이름: 제출 설명에서 `lgbm`, `kumo+catboost`, `exaone@knn1000` 처럼 모델 이름으로 해석되는 첫 토큰.
         RETIRED 모델(limix2)이 들어간 제출은 순위 집계·자동 제출에서 뺌.
실행: 제출을 만든 실행의 CV 와 시각(run 열 = results.csv timestamp)을 같이 기록 → km board 가 실행 단위로 LB 를 붙임.
     설명의 `run <시각>` (km submit 기본) → `<n>-fold CV <점수>` → 제출 시각 전 마지막 실행 순으로 찾음.
     못 찾으면 이전 leaderboard.csv 에서 찾은 값을 제출 ref 로 유지. --submit 도 이미 제출한 '실행' 은 다시 내지 않음.
"""

from __future__ import annotations

import argparse
import functools
import io
import re
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from km.core.competition import COMP_DIR, load_competition
from km.core.metrics import METRICS
from km.core.results import RESULTS, made_test, parse_utc, read_results, test_info
from km.leaderboard.submit import make_submission, parse_message, push
from km.models import REGISTRY, RETIRED, base_name


@functools.cache
def kaggle_api():
    """인증한 KaggleApi (kaggle 패키지, kagglesdk 클라이언트와 username 을 얻음). 설치·인증 문제 → RuntimeError."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api
    except (Exception, SystemExit) as e:  # 인증 정보가 없으면 authenticate() 가 exit(1)
        raise RuntimeError(f"Kaggle 인증 실패: {e!r}") from e


def me() -> str:
    user = kaggle_api().config_values.get("username")
    if not user:
        raise RuntimeError("Kaggle username 을 찾지 못함")
    return user


@functools.cache
def sdk():
    """kagglesdk 대회 API 클라이언트. km lb 한 번에 세션 하나 (세션은 프로세스가 끝날 때까지 씀)."""
    try:
        client = kaggle_api().build_kaggle_client().__enter__()
        # kagglesdk 는 요청에 타임아웃을 안 줌 → 응답이 멈추면 km lb 전체가 멈추지 않도록 (예전 CLI 호출은 600초).
        # 비공개 속성이라 kagglesdk 가 바뀌면 여기서 깨짐 → 이 try 안에서 RuntimeError 로
        session = client.http_client()._session
        session.send = functools.partial(session.send, timeout=300)
    except Exception as e:
        raise RuntimeError(f"kagglesdk 준비 실패: {e!r}") from e
    return client.competitions.competition_api_client


def sdk_call(what: str, fn, req):
    """429 (요청 제한, 연속 조회 중 자주 남) 는 잠깐 쉬고 다시. 그 밖의 네트워크·HTTP 오류 (Rules 미동의 등) 는 RuntimeError → main 이 이 대회만 건너뜀."""
    import requests

    for attempt in range(5):
        try:
            return fn(req)
        except requests.exceptions.HTTPError as e:
            if e.response is None or e.response.status_code != 429 or attempt == 4:
                raise RuntimeError(f"{what} 실패: {e}") from e
            time.sleep(10 * (attempt + 1))
        except requests.exceptions.RequestException as e:  # 연결 끊김·타임아웃 등
            raise RuntimeError(f"{what} 실패: {e}") from e


def pages(what: str, fn, req_cls, comp: str):
    """페이지 끝까지 응답을 하나씩 (200개씩). 제출 기록·리더보드 조회 응답은 둘 다 submissions + next_page_token.
    같은 토큰이 다시 오거나 빈 페이지에 토큰이 붙어 오면 RuntimeError (같은 페이지를 끝없이 다시 받지 않도록)."""
    seen, token = set(), None
    while True:
        req = req_cls()
        req.competition_name = comp
        req.page_size = 200
        if token:
            req.page_token = token
        resp = sdk_call(what, fn, req)
        yield resp
        token = resp.next_page_token
        if not token:
            return
        if token in seen or not resp.submissions:
            raise RuntimeError(f"{what} 실패: 페이지가 끝나지 않음")
        seen.add(token)


def submissions(comp: str) -> list[dict]:
    """제출 기록 전부 (페이지 끝까지). 읽지 못하면 빈 목록 대신 예외 → `--submit` 이 이미 낸 모델을 다시 내지 않도록.
    키는 예전 `kaggle competitions submissions --format json` 과 같음 + teamName. date 는 UTC naive isoformat.

    (2026-10-23: CLI 출력의 JSON 앞에 버전 경고 줄이 붙어 기록이 늘 빈 목록 → 이미 낸 모델 21개를 다시 제출. 지금은 SDK 응답을 바로 씀)
    """
    from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest

    out = []
    for resp in pages("제출 기록 조회", sdk().list_submissions, ApiListSubmissionsRequest, comp):
        for s in resp.submissions or []:
            if s.date is None:
                raise RuntimeError(f"제출 {s.ref} 의 시각이 없음")
            out.append(dict(ref=s.ref, fileName=s.file_name, date=s.date.isoformat(), description=s.description,
                            status=s.status.name, publicScore=s.public_score, privateScore=s.private_score,
                            teamName=s.team_name))
    return out


def public_leaderboard(comp: str) -> pd.DataFrame:
    """public 리더보드 다운로드 (zip 안의 CSV: TeamId, TeamName, ..., Score, TeamMemberUserNames). 진행 중 대회에만 부름."""
    from kagglesdk.competitions.types.competition_api_service import ApiDownloadLeaderboardRequest

    req = ApiDownloadLeaderboardRequest()
    req.competition_name = comp
    data = sdk_call("리더보드 다운로드", lambda r: sdk().download_leaderboard(r).content, req)
    try:  # pandas 의 빈 파일·파싱·인코딩 오류는 모두 ValueError
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open(next(n for n in z.namelist() if n.endswith(".csv"))) as f:
                lb = pd.read_csv(f, encoding="utf-8-sig")
    except (zipfile.BadZipFile, StopIteration, ValueError) as e:
        raise RuntimeError(f"리더보드 파일을 읽지 못함: {e!r}") from e
    if missing := {"Score", "TeamMemberUserNames"} - set(lb.columns):
        raise RuntimeError(f"리더보드 파일에 열이 없음: {sorted(missing)}")
    return lb


def deadline(comp: str) -> datetime:
    """대회 마감 시각 (UTC naive). kagglesdk 가 UTC 'Z' 를 떼고 준 naive datetime (타임존이 붙어 와도 parse_utc 로 맞춤)."""
    from kagglesdk.competitions.types.competition_api_service import ApiGetCompetitionRequest

    req = ApiGetCompetitionRequest()
    req.competition_name = comp
    at = sdk_call("대회 정보 조회", sdk().get_competition, req).deadline
    if at is None:
        raise RuntimeError("대회 마감일을 모름")
    return parse_utc(at.isoformat())


def private_leaderboard(comp: str) -> pd.DataFrame:
    """리더보드 조회 API 의 기본 응답 전 페이지 (TeamId, TeamName, Score). private 이 공개됐으면 private, 아니면 public. 종료 대회에만 부름."""
    from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

    rows = []
    for resp in pages("리더보드 조회", sdk().get_leaderboard, ApiGetLeaderboardRequest, comp):
        try:
            rows += [(s.team_id, s.team_name, float(s.score)) for s in resp.submissions or [] if s.score not in (None, "")]
        except ValueError as e:
            raise RuntimeError(f"리더보드 점수를 읽지 못함: {e}") from e
    return pd.DataFrame(rows, columns=["TeamId", "TeamName", "Score"])


def model_of(sub: dict) -> str:
    for tok in re.findall(r"[\w.@+-]+", sub["description"] or ""):
        if all(base_name(p) in REGISTRY or base_name(p) in RETIRED for p in tok.split("+")):
            return tok
    return Path(sub["fileName"]).stem  # 설명에 모델 이름이 없으면 파일 이름 ('+', '@' 는 Kaggle 이 지움)


def is_retired(model: str) -> bool:
    """RETIRED 모델이 들어간 모델·앙상블 (예: limix2+kumo) → 리더보드 집계와 자동 제출에서 뺌."""
    return any(base_name(p) in RETIRED for p in model.split("+"))


def prediction_runs(comp, results: pd.DataFrame) -> pd.DataFrame:
    """제출과 연결할 후보: 전체 fold 실행 중 테스트 예측을 만든 것 (탐색 실험·--no-test 재실행 제외), 시각 순.
    results: read_results() 의 results.csv 전체. run_at = timestamp 를 parse_utc 로 한 번만 읽은 값 (submission_run 이 씀).
    시각을 못 읽는 행은 경고하고 후보에서 뺌 (그 행 하나 때문에 대회 전체를 건너뛰지 않음)."""
    runs = results[(results.comp == comp.name) & (results.n_folds == comp.cv["n_splits"])]
    runs = runs[made_test(runs)]
    at = []
    for t in runs.timestamp:
        try:
            at.append(parse_utc(t))
        except RuntimeError as e:
            print(f"{comp.name}: results.csv 행을 제출 연결 후보에서 뺌 ({e})")
            at.append(None)
    runs = runs.assign(run_at=at)
    return runs[runs.run_at.notna()].sort_values("run_at")


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
    if info.run:  # 적힌 실행이 없으면 (지워짐 등) 추측하지 않음
        hit = runs[runs.run_at == parse_utc(info.run)]
        return (float(hit.cv.iloc[-1]), hit.timestamp.iloc[-1]) if len(hit) else (float(info.cv) if cv_ok else None, None)
    before = runs[runs.run_at <= parse_utc(sub["date"])]
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
    # 종료 대회는 private 기준. 우리 제출에 private 점수가 없으면 (private 비공개) public 기준으로 둠
    has_private = any(s["privateScore"] not in (None, "") for s in subs)
    closed_at = deadline(comp_name) if has_private else None
    basis = "private" if closed_at and closed_at < datetime.now(timezone.utc).replace(tzinfo=None) else "public"
    if basis == "private":
        priv = private_leaderboard(comp_name)
        if priv.empty:
            raise RuntimeError("리더보드 조회 결과가 비어 있음")
        # 마감 뒤 제출 (late submission) 은 리더보드에 안 올라감 (2026-10-27 확인) → 마감 전 제출이 있을 때만 우리 팀을 찾아 뺌.
        # 조회 API 에는 멤버가 없어 팀 이름으로 찾음. 못 찾거나 같은 이름이 여럿이면 순위가 하나 틀리니 건너뜀
        mine = pd.Series(False, index=priv.index)
        if any(parse_utc(s["date"]) <= closed_at for s in subs):
            team = {s["teamName"] for s in subs}
            mine = priv.TeamName.isin(team)
            if mine.sum() != 1:
                raise RuntimeError(f"리더보드에서 우리 팀 {sorted(team)} 이 {mine.sum()}개")
        others = priv.loc[~mine, "Score"].to_numpy()
    else:
        lb = public_leaderboard(comp_name)
        mine = lb.TeamMemberUserNames.fillna("").str.split(",").apply(lambda u: user in [x.strip() for x in u])
        others = lb.loc[~mine, "Score"].to_numpy()
    runs = prediction_runs(comp, results)
    rows = []
    for s in subs:
        score = s["privateScore"] if basis == "private" else s["publicScore"]
        # 종료 대회인데 private 점수가 없는 제출 → 순위 없이 행만 남김 (실행 연결은 다음 km lb 의 known 으로 유지)
        rank = None if score in (None, "") else int(((others > float(score)) if higher else (others < float(score))).sum()) + 1
        model = model_of(s)
        if is_retired(model):
            continue
        cv, run = submission_run(comp, s, runs[runs.model == model])
        if run is None and str(s["ref"]) in (known or {}):
            cv, run = known[str(s["ref"])]
        rows.append(dict(comp=comp_name, model=model, metric=comp.metric, basis=basis, cv=cv,
                         public=float(s["publicScore"]),
                         private=float(s["privateScore"]) if s["privateScore"] not in (None, "") else None,
                         rank=rank, teams=len(others) + 1, top_pct=None if rank is None else round(100 * rank / (len(others) + 1), 1),
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
        except RuntimeError as e:  # 제출 기록을 못 읽음 (Rules 미동의 등) → 이 대회는 제출·집계 모두 건너뜀
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

