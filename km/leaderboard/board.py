"""results.csv · leaderboard.csv → 모델별 성능·실행 시간 리더보드 (docs/leaderboard.md, docs/leaderboard.html)

예: uv run km board            # docs/leaderboard.md 와 정렬·필터·차트가 있는 docs/leaderboard.html 다시 생성
    uv run km board -o -       # 파일 대신 화면에 출력 (md 만)

`km run`/`km databricks` 로 결과가 늘거나 `km lb` 로 순위가 바뀌면 다시 생성.
s/1K 계산에 대회별 학습·테스트 행 수가 필요 → 로컬 데이터(`km download`)나 저장된 예측(oof/test.npy)에서 읽음.
둘 다 없는 대회는 s/1K 를 비워 두고 경고만 출력.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from km.core.competition import COMP_DIR, ROOT, load_competition
from km.core.metrics import METRICS
from km.core.results import OUTPUTS, RESULTS, is_exploration, made_test
from km.models import REGISTRY

OUT = ROOT / "docs" / "leaderboard.md"
TEMPLATE = Path(__file__).with_name("board.html")  # script#data 의 `__DATA__` 에 JSON 을 넣음
GBDT = {name for name, (module, _) in REGISTRY.items() if module == "km.models.gbdt"}
TITLES = {  # 표 제목용 짧은 이름 (없으면 대회 이름 그대로)
    "restaurant-revenue-prediction": "Restaurant", "titanic": "Titanic",
    "house-prices-advanced-regression-techniques": "House Prices", "bioresponse": "Bioresponse",
    "mercedes-benz-greener-manufacturing": "Mercedes", "spaceship-titanic": "Spaceship",
    "playground-series-s4e2": "S4E2", "amazon-employee-access-challenge": "Amazon",
    "otto-group-product-classification-challenge": "Otto", "GiveMeSomeCredit": "GiveMeSomeCredit",
    "allstate-claims-severity": "Allstate", "playground-series-s6e1": "S6E1", "playground-series-s6e10": "S6E10",
}
# results.csv 의 where 가 databricks 인 실행 중 이 날짜 전은 A10G 노드 (GBDT 도 같은 노드), 이후는 H100 (GBDT 는 databricks-cpu)
H100_SINCE = "2026-10-23"
BASIS_LABEL = {"public": "public (진행 중)", "private": "private (종료)"}  # leaderboard.csv 의 basis 열 (km lb 참고)


def row_counts(name: str) -> tuple[int, int] | None:
    """(학습 행, 테스트 행). 로컬 데이터 → 저장된 예측 순으로 찾고, 없으면 None."""
    comp = load_competition(name)
    train, test = comp.data_dir / comp.train_file, comp.data_dir / comp.test_file
    if train.exists() and test.exists():
        return len(pd.read_csv(train, usecols=[0])), len(pd.read_csv(test, usecols=[0]))
    for d in (OUTPUTS / name).glob("*/"):
        if (d / "oof.npy").exists() and (d / "test.npy").exists():
            return len(np.load(d / "oof.npy", mmap_mode="r")), len(np.load(d / "test.npy", mmap_mode="r"))
    return None


def env(r) -> str:
    where = r["where"]
    if pd.isna(where):
        return "—"
    if where == "databricks-cpu":
        return "DBX CPU4"
    if where == "databricks":
        return "DBX A10G" if r["timestamp"] < H100_SINCE else "DBX H100"
    return {"kaggle": "Kaggle T4", "colab": "Colab T4", "local": "로컬"}.get(where, where)


def label(model: str, en: bool = False) -> str:
    """`kumo@full` → `kumo (ctx 전체)`, `kumo@rand200000` → `kumo (ctx 20만)`, `exaone@knn1000_g8_imp` → `exaone (kNN 1000)`.
    en=True 면 `kumo (ctx full)`, `kumo (ctx 200K)` (HTML 영어 화면용)."""
    base, _, spec = model.partition("@")
    if not spec:
        return base
    mode, k = re.match(r"(full|rand|knn|part)(\d*)", spec).groups()
    if mode == "full":
        return f"{base} (ctx {'full' if en else '전체'})"
    size = (f"{int(k) // 1000}K" if en else f"{int(k) // 10000}만") if int(k) % 10000 == 0 else k
    return f"{base} ({ {'rand': 'ctx', 'knn': 'kNN', 'part': 'part'}[mode] } {size})"


def load() -> tuple[pd.DataFrame, dict]:
    res = pd.read_csv(RESULTS / "results.csv")
    lb = pd.read_csv(RESULTS / "leaderboard.csv")  # 제출마다 한 행, 대회·순위 순 정렬
    for col in ("private", "basis"):  # 이 열이 생기기 전 km lb 가 쓴 파일 → public 기준으로 봄
        if col not in lb:
            lb[col] = "public" if col == "basis" else np.nan
    comps = sorted(p.name for p in COMP_DIR.iterdir() if (p / "config.yaml").exists() and not p.name.startswith("_"))
    d = res[res.comp.isin(comps) & (res.n_folds == 5) & res.cv.notna() & res.seconds.notna()]
    d = d[~d.model.str.contains(r"\+|llm|sap_rpt")]  # 앙상블, LLM 결정 모델, sap_rpt
    # ctx: 컨텍스트 탐색 실험 (테스트 예측 없음). llm·sap: 결정 모델 실험 중 같이 다시 돌린 GBDT
    d = d[~is_exploration(d.note)]
    d = d[made_test(d)]  # 테스트 예측을 만든 실행만 → --no-test 재실행이 제출한 예측의 LB 를 가리지 않음
    d = d.sort_values("timestamp").drop_duplicates(["comp", "model"], keep="last")

    sizes = {}
    for c in d.comp.unique():
        sizes[c] = row_counts(c)
        if sizes[c] is None:
            print(f"{c}: 학습·테스트 행 수를 몰라 s/1K 생략 (`uv run km download {c}` 후 다시 실행)", file=sys.stderr)
    d["env"] = d.apply(env, axis=1)
    d["s_per_1k"] = [s / (n[0] + 5 * n[1]) * 1000 if (n := sizes[c]) else np.nan for c, s in zip(d.comp, d.seconds)]
    d["family"] = np.where(d.model.isin(GBDT), "GBDT", "TFM")
    d["label"] = d.model.map(label)
    d["base"] = d.label.str.partition(" (")[0]
    d["better"] = [cv if METRICS[m][1] else -cv for cv, m in zip(d.cv, d.metric)]
    d["cv_rank"] = d.groupby("comp").better.rank(ascending=False, method="min").astype(int)
    # 제출은 그 제출을 만든 실행(leaderboard.csv 의 run = results.csv 의 timestamp)에만 붙임 → 같은 이름으로 다시 돌린 실행에
    # 예전 제출 LB 가 붙지 않음. 같은 실행을 여러 번 냈으면 최고 순위 (leaderboard.csv 가 대회·순위 순)
    run = (lb["run"] if "run" in lb else pd.Series(None, index=lb.index)).astype(object)  # 비어 있어도 문자열 열로
    single = ~lb.model.astype(str).str.contains(r"\+")
    if (missing := int((single & run.isna()).sum())):
        print(f"leaderboard.csv: 실행(run)을 모르는 단일 모델 제출 {missing}건은 LB 를 붙이지 않음 (`km lb` 로 다시 집계)",
              file=sys.stderr)
    known = lb[run.notna()].assign(timestamp=run[run.notna()]).drop_duplicates(["comp", "model", "timestamp"])
    d = d.merge(known[["comp", "model", "timestamp", "public", "private", "basis", "rank", "teams", "top_pct"]],
                on=["comp", "model", "timestamp"], how="left")
    d["lb_score"] = d.private.where(d.basis == "private", d.public)  # 순위와 같은 기준 점수 (종료 대회는 private)
    return d, sizes


def fmt_pct(p: float) -> str:
    return "<0.1%" if p < 0.1 else f"{p:.1f}%"


def fmt_time(s: float) -> str:
    return f"{s:,.0f}s"


def fmt_num(v: float) -> str:
    return f"{v:,.2f}" if abs(v) > 1000 else f"{v:.5g}"


def best_per_base(d: pd.DataFrame) -> pd.DataFrame:
    """컨텍스트 변형은 기본 모델로 합치고 대회마다 CV 가 가장 좋은 설정 한 행만. cv_rank 는 그 안에서 다시 매김."""
    b = d.sort_values("better", ascending=False).drop_duplicates(["comp", "base"]).copy()
    b["cv_rank"] = b.groupby("comp").better.rank(ascending=False, method="min").astype(int)
    b["n_models"] = b.groupby("comp").model.transform("size")
    return b


def summary_rows(d: pd.DataFrame) -> pd.DataFrame:
    """모델별 종합, 평균 CV 순위 순. 점수·시간은 대회별 최고 CV 설정 행 기준.
    최고 LB 는 제출한 모든 설정 중 최고 (최고 CV 설정이 미제출이면 그 행만으로는 LB 가 비거나 나빠 보임)."""
    rows = []
    for m, g in best_per_base(d).groupby("base"):
        best_lb = d.loc[d.base == m, "top_pct"].min()  # 제출한 모든 설정 중
        rows.append(dict(model=m, family=g.family.iloc[0], n=len(g), wins=int((g.cv_rank == 1).sum()),
                         pct=((g.cv_rank - 1) / (g.n_models - 1).clip(lower=1)).mean(), lb=best_lb,
                         # 최고 LB 가 최고 CV 설정이 아닌 행에서 옴 (최고 CV 설정이 미제출이거나 순위가 더 낮음)
                         lb_other=pd.notna(best_lb) and not g.top_pct.min() <= best_lb,
                         s1k=g[g.env.isin(["DBX H100", "DBX CPU4"])].s_per_1k.median(), s1k_all=g.s_per_1k.median(),
                         ctx=int(g.model.str.contains("@").sum())))
    return pd.DataFrame(rows).sort_values("pct")


def summary(d: pd.DataFrame) -> list[str]:
    out = ["| 모델 | 계열 | 대회 수 | CV 1위 | 평균 CV 순위 | 최고 LB (상위 %) | s/1K (DBX) | s/1K (전체) | 컨텍스트 변형 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in summary_rows(d).itertuples():
        lb = fmt_pct(r.lb) + ("*" if r.lb_other else "") if pd.notna(r.lb) else "—"
        cells = [r.model, r.family, r.n, r.wins, f"{r.pct:.2f}", lb,
                 f"{r.s1k:.2f}" if pd.notna(r.s1k) else "—", f"{r.s1k_all:.2f}" if pd.notna(r.s1k_all) else "—",
                 r.ctx or ""]
        out.append("| " + " | ".join(map(str, cells)) + " |")
    return out


def comp_basis(d: pd.DataFrame) -> dict:
    """대회 → LB 기준 (public·private, km lb 의 basis 열). 제출이 없는 대회는 빠짐."""
    return {c: b.iloc[-1] for c, b in d.dropna(subset=["basis"]).groupby("comp").basis}


def per_comp(d: pd.DataFrame, sizes: dict) -> list[str]:
    bases = comp_basis(d)
    out = []
    for comp in sorted(d.comp.unique(), key=lambda c: (sizes[c] or (0, 0))[0]):
        g = d[d.comp == comp].sort_values("cv_rank")
        metric = g.metric.iloc[0]
        rows = f"학습 {sizes[comp][0]:,}행 · 테스트 {sizes[comp][1]:,}행 · " if sizes[comp] else ""
        basis = f" · LB 기준 {BASIS_LABEL.get(b, b)}" if (b := bases.get(comp)) else ""
        out += [f"### {TITLES.get(comp, comp)}\n", f"{rows}{metric} {'↑' if METRICS[metric][1] else '↓'}{basis}\n",
                f"| # | 모델 | CV {metric} | ± std | LB 점수 | LB 순위 | 시간 | s/1K | 환경 |",
                "|---|---|---|---|---|---|---|---|---|"]
        best = g.better.max()
        for r in g.itertuples():
            name = f"**{r.label}**" if r.better == best else r.label
            rank = "—" if pd.isna(r.rank) else f"{int(r.rank):,} / {int(r.teams):,} ({fmt_pct(r.top_pct)})"
            cells = [r.cv_rank, f"{name} 🌳" if r.family == "GBDT" else name,
                     f"{r.cv:,.1f}" if abs(r.cv) > 1000 else f"{r.cv:.4f}",
                     f"{r.cv_std:,.1f}" if abs(r.cv) > 1000 else f"{r.cv_std:.4f}",
                     "—" if pd.isna(r.lb_score) else fmt_num(r.lb_score), rank, fmt_time(r.seconds),
                     "—" if pd.isna(r.s_per_1k) else f"{r.s_per_1k:.2f}", r.env]
            out.append("| " + " | ".join(map(str, cells)) + " |")
        out.append("")
    return out


HEADER = """\
# 리더보드 (모델별 성능 · 실행 시간)

`uv run km board` 로 `results/results.csv`·`results/leaderboard.csv` 에서 생성 ({date}). 직접 고치지 말 것. 5-fold, 튜닝 없는 기본 설정.

- **시간** = 5-fold 합계 (학습 + 검증·테스트 예측). TFM 은 학습 루프가 없어 거의 전부 추론(컨텍스트 인코딩 + 예측) 시간.
- **s/1K** = 시간 ÷ 예측 행 수 × 1000. 예측 행 = 학습 행(OOF) + 테스트 행 × 5 (fold 마다 테스트 전체 예측). TabArena 의 predict s/1K 와 비슷한 단위.
- **환경이 다르면 시간 비교 불가**: DBX H100 (TFM, 10-23~) · DBX CPU4 (GBDT, serverless 4코어, 10-23~) · DBX A10G (10-22, GBDT 도 같은 노드) · Kaggle T4.
- **LB 순위**는 `km lb` 집계. 진행 중 대회는 public 점수 기준, 종료 대회는 private 점수 기준 (대회 제목 옆 `LB 기준`). 종료 대회는 늦은 제출이라 실제 순위표에는 없음. — = 미제출.
- 작은 대회(수천 행 이하)는 모델 로드·전처리 같은 고정 비용이 커서 s/1K 가 부풀려짐 → 큰 대회 값이 모델 속도에 더 가까움.
- 앙상블·LLM 결정 모델·sap_rpt, 컨텍스트 탐색 실험(results.csv note 가 `ctx ` 로 시작, 테스트 예측 없음)은 제외.
  `(kNN 1000)` 은 포함: S4E2·Spaceship 에서 kumo·exaone·tabpfn_v2 는 전체 컨텍스트로 실패해 kNN 1000 행 컨텍스트가 유일한 5-fold·제출 결과.
- TFM 컨텍스트 상한: 대회 config 의 `tfm_max_rows` ({capped}) 는 무작위 샘플, 나머지 대회는 상한 없이 학습 fold 전체.
  `(ctx 전체)` / `(ctx 20만)` = 그 대회들에서 상한을 풀거나 20만 행으로 올린 실행 (`@full` / `@rand200000`).

## 종합

평균 CV 순위 = 대회마다 (순위−1)/(모델 수−1) 의 평균 (0 = 항상 1위, 1 = 항상 꼴찌). 실행한 대회가 모델마다 달라 대략적인 비교.
모델마다 대회별로 CV 가 가장 좋은 설정 하나만 셈 (`컨텍스트 변형` 열 = 그중 `@` 설정 수). s/1K 도 그 행의 값, 최고 LB 는 제출한 모든 설정 중 최고 (`*` = 최고 CV 설정이 아닌 설정의 제출).
s/1K 중앙값: `DBX` = 2026-10-23 이후 Databricks 실행만 (TFM H100, GBDT CPU 4코어), `전체` = 환경 섞임.

**해석 주의**
- **조건이 같지 않을 수 있음**: 상한이 걸린 대회에서 `@full`·`@rand` 로 다시 돌린 모델만 큰 컨텍스트를 씀 ({rerun}).
  다시 돌리지 않은 모델은 상한 그대로라, 그 대회의 순위 일부는 모델 차이가 아니라 컨텍스트 차이.
- **사전학습 데이터 중복 가능성**: Otto·GiveMeSomeCredit·Amazon 등은 OpenML 등에 공개된 데이터라 TFM 사전학습에 학습 데이터가 들어갔을 수 있음 → CV 는 부풀었을 수 있음.
  테스트 라벨은 공개된 적이 없어 LB 점수는 영향 없음. 확인은 안 함.
"""


def rerun_status(d: pd.DataFrame, caps: dict) -> str:
    """상한(caps: 대회 → tfm_max_rows)보다 큰 컨텍스트(`@full`, `@rand<k>` k > 상한)의 5-fold 결과가 있는 대회 수, 모델별
    (예: `kumo 6/6, tabpfn_v2 3/6 (Amazon, Otto)`)."""
    def bigger(model: str, comp: str) -> bool:
        m = re.search(r"@(full|rand(\d+))", model)  # 숫자 없는 `@rand` 등은 큰 컨텍스트로 안 셈
        return bool(m) and (m.group(1) == "full" or int(m.group(2)) > caps[comp])

    capped = list(caps)
    big = d[[c in caps and bigger(m, c) for m, c in zip(d.model, d.comp)]]
    parts = []
    for base, g in big.groupby("base"):
        done = sorted(g.comp.unique(), key=lambda c: TITLES.get(c, c))
        names = f" ({', '.join(TITLES.get(c, c) for c in done)})" if len(done) < len(capped) else ""
        parts.append(f"{base} {len(done)}/{len(capped)}{names}")
    return ", ".join(parts) or "없음"


def render(d: pd.DataFrame, sizes: dict) -> str:
    comps = sorted(d.comp.unique(), key=lambda c: (sizes[c] or (0, 0))[0])
    caps = {c: n for c in comps if (n := load_competition(c).tfm_max_rows)}  # 대회 → TFM 컨텍스트 상한
    by_cap = {}  # 상한 → 대회들
    for c, n in caps.items():
        by_cap.setdefault(n, []).append(TITLES.get(c, c))
    capped = "; ".join(f"{n:,}행: {', '.join(cs)}" for n, cs in by_cap.items()) or "없음"
    head = HEADER.format(date=pd.Timestamp.now().strftime("%Y-%m-%d"), capped=capped, rerun=rerun_status(d, caps))
    return "\n".join([head, *summary(d), "", "## 대회별\n", *per_comp(d, sizes)])


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", force_ascii=False))  # NaN → null


def render_html(d: pd.DataFrame, sizes: dict) -> str:
    """docs/leaderboard.html: md 와 같은 표를 JSON 으로 넣고 정렬·필터·차트는 템플릿의 JS 가 그림 (외부 의존성 없음)."""
    comps = sorted(d.comp.unique(), key=lambda c: (sizes[c] or (0, 0))[0])
    # base_rank = 종합·히트맵용 순위 (기본 모델끼리, 대회별 최고 설정 행에만). cv_rank 는 설정별 순위
    best = best_per_base(d)[["comp", "model", "cv_rank"]].rename(columns={"cv_rank": "base_rank"})
    runs = d.merge(best, on=["comp", "model"], how="left").assign(label_en=lambda x: x.model.map(lambda m: label(m, en=True)))
    basis = comp_basis(d)
    runs = runs[["comp", "model", "label", "label_en", "base", "family", "cv", "cv_std", "cv_rank", "base_rank", "lb_score", "rank",
                 "teams", "top_pct", "seconds", "s_per_1k", "env"]]
    caps = {c: n for c in comps if (n := load_competition(c).tfm_max_rows)}
    data = dict(
        date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        comps=[dict(id=c, title=TITLES.get(c, c), metric=(m := d[d.comp == c].metric.iloc[0]), higher=METRICS[m][1],
                    train=sizes[c] and sizes[c][0], test=sizes[c] and sizes[c][1], cap=caps.get(c), basis=basis.get(c)) for c in comps],
        rerun=rerun_status(d, caps),  # md 해석 주의와 같은 문구 (예: `kumo 6/6, tabicl 3/6 (...)`)
        runs=records(runs.sort_values(["comp", "cv_rank"])),
        summary=records(summary_rows(d)),
    )
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # <script> 안에 넣으므로 </script> 방지
    return TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload, 1)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km board")
    ap.add_argument("-o", "--out", default=str(OUT), help="md 출력 경로 (- 면 화면에 출력). 기본: docs/leaderboard.md")
    ap.add_argument("--html", help="HTML 출력 경로 (- 면 생략). 기본: md 와 같은 위치·이름의 .html (-o - 면 생략)")
    args = ap.parse_args(argv)
    html = args.html or ("-" if args.out == "-" else str(Path(args.out).with_suffix(".html")))
    d, sizes = load()
    text = render(d, sizes)
    outs = ([] if args.out == "-" else [(Path(args.out), text)]) + ([] if html == "-" else [(Path(html), render_html(d, sizes))])
    for out, body in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        print(f"저장: {out}", file=sys.stderr if args.out == "-" else sys.stdout)
    if args.out == "-":
        print(text)
