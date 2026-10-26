"""results.csv · leaderboard.csv → 모델별 성능·실행 시간 리더보드 (docs/leaderboard.md)

예: uv run km board            # docs/leaderboard.md 다시 생성
    uv run km board -o -       # 파일 대신 화면에 출력

`km run`/`km databricks` 로 결과가 늘거나 `km lb` 로 순위가 바뀌면 다시 생성.
s/1K 계산에 대회별 학습·테스트 행 수가 필요 → 로컬 데이터(`km download`)나 저장된 예측(oof/test.npy)에서 읽음.
둘 다 없는 대회는 s/1K 를 비워 두고 경고만 출력.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from km.competition import COMP_DIR, ROOT, load_competition
from km.metrics import METRICS
from km.models import REGISTRY, RETIRED
from km.results import RESULTS

OUT = ROOT / "docs" / "leaderboard.md"
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


def row_counts(name: str) -> tuple[int, int] | None:
    """(학습 행, 테스트 행). 로컬 데이터 → 저장된 예측 순으로 찾고, 없으면 None."""
    comp = load_competition(name)
    train, test = comp.data_dir / comp.train_file, comp.data_dir / comp.test_file
    if train.exists() and test.exists():
        return len(pd.read_csv(train, usecols=[0])), len(pd.read_csv(test, usecols=[0]))
    for d in (RESULTS / name).glob("*/"):
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


def label(model: str) -> str:
    """`kumo@full` → `kumo (ctx 전체)`, `kumo@rand200000` → `kumo (ctx 20만)`, `exaone@knn1000_g8_imp` → `exaone (kNN 1000)`."""
    base, _, spec = model.partition("@")
    base = f"{base}†" if base in RETIRED else base
    if not spec:
        return base
    mode, k = re.match(r"(full|rand|knn|part)(\d*)", spec).groups()
    if mode == "full":
        return f"{base} (ctx 전체)"
    size = f"{int(k) // 10000}만" if int(k) % 10000 == 0 else k
    return f"{base} ({ {'rand': 'ctx', 'knn': 'kNN', 'part': 'part'}[mode] } {size})"


def load() -> tuple[pd.DataFrame, dict]:
    res = pd.read_csv(RESULTS / "results.csv")
    lb = pd.read_csv(RESULTS / "leaderboard.csv").drop_duplicates(["comp", "model"])  # 순위 순 정렬 → 같은 모델은 최고 순위
    comps = sorted(p.name for p in COMP_DIR.iterdir() if (p / "config.yaml").exists() and not p.name.startswith("_"))
    d = res[res.comp.isin(comps) & (res.n_folds == 5) & res.cv.notna() & res.seconds.notna()]
    d = d[~d.model.str.contains(r"\+|llm|sap_rpt")]  # 앙상블, LLM 결정 모델, sap_rpt
    # ctx: 컨텍스트 탐색 실험 (테스트 예측 없음). llm·sap: 결정 모델 실험 중 같이 다시 돌린 GBDT
    d = d[~d.note.fillna("").str.match(r"ctx |llm|sap")]
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
    d = d.merge(lb[["comp", "model", "public", "rank", "teams", "top_pct"]], on=["comp", "model"], how="left")
    return d, sizes


def fmt_pct(p: float) -> str:
    return "<0.1%" if p < 0.1 else f"{p:.1f}%"


def fmt_time(s: float) -> str:
    return f"{s:,.0f}s" if s < 3600 else f"{s / 3600:.1f}h"


def fmt_num(v: float) -> str:
    return f"{v:,.2f}" if abs(v) > 1000 else f"{v:.5g}"


def summary(d: pd.DataFrame) -> list[str]:
    """모델별 종합. 컨텍스트 변형은 기본 모델로 합치고 대회마다 CV 가 가장 좋은 설정 한 행만 (점수·LB·시간 모두 그 행)."""
    b = d.sort_values("better", ascending=False).drop_duplicates(["comp", "base"]).copy()
    b["cv_rank"] = b.groupby("comp").better.rank(ascending=False, method="min").astype(int)
    b["n_models"] = b.groupby("comp").model.transform("size")
    rows = []
    for m, g in b.groupby("base"):
        rows.append(dict(model=m, family=g.family.iloc[0], n=len(g), wins=int((g.cv_rank == 1).sum()),
                         pct=((g.cv_rank - 1) / (g.n_models - 1).clip(lower=1)).mean(), lb=g.top_pct.min(),
                         s1k=g[g.env.isin(["DBX H100", "DBX CPU4"])].s_per_1k.median(), s1k_all=g.s_per_1k.median(),
                         ctx=int(g.model.str.contains("@").sum())))
    out = ["| 모델 | 계열 | 대회 수 | CV 1위 | 평균 CV 순위 | 최고 LB (상위 %) | s/1K (DBX) | s/1K (전체) | 컨텍스트 변형 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in pd.DataFrame(rows).sort_values("pct").itertuples():
        cells = [r.model, r.family, r.n, r.wins, f"{r.pct:.2f}", fmt_pct(r.lb) if pd.notna(r.lb) else "—",
                 f"{r.s1k:.2f}" if pd.notna(r.s1k) else "—", f"{r.s1k_all:.2f}" if pd.notna(r.s1k_all) else "—",
                 r.ctx or ""]
        out.append("| " + " | ".join(map(str, cells)) + " |")
    return out


def per_comp(d: pd.DataFrame, sizes: dict) -> list[str]:
    out = []
    for comp in sorted(d.comp.unique(), key=lambda c: (sizes[c] or (0, 0))[0]):
        g = d[d.comp == comp].sort_values("cv_rank")
        metric = g.metric.iloc[0]
        rows = f"학습 {sizes[comp][0]:,}행 · 테스트 {sizes[comp][1]:,}행 · " if sizes[comp] else ""
        out += [f"### {TITLES.get(comp, comp)}\n", f"{rows}{metric} {'↑' if METRICS[metric][1] else '↓'}\n",
                f"| # | 모델 | CV {metric} | ± std | LB public | LB 순위 | 시간 | s/1K | 환경 |",
                "|---|---|---|---|---|---|---|---|---|"]
        best = g.better.max()
        for r in g.itertuples():
            name = f"**{r.label}**" if r.better == best else r.label
            rank = "—" if pd.isna(r.rank) else f"{int(r.rank):,} / {int(r.teams):,} ({fmt_pct(r.top_pct)})"
            cells = [r.cv_rank, f"{name} 🌳" if r.family == "GBDT" else name,
                     f"{r.cv:,.1f}" if abs(r.cv) > 1000 else f"{r.cv:.4f}",
                     f"{r.cv_std:,.1f}" if abs(r.cv) > 1000 else f"{r.cv_std:.4f}",
                     "—" if pd.isna(r.public) else fmt_num(r.public), rank, fmt_time(r.seconds),
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
- **LB 순위**는 `km lb` 집계 (public 점수 기준 위치. 종료 대회는 늦은 제출이라 실제 순위표에는 없음). — = 미제출. † = 실행 목록에서 뺀 모델.
- 작은 대회(수천 행 이하)는 모델 로드·전처리 같은 고정 비용이 커서 s/1K 가 부풀려짐 → 큰 대회 값이 모델 속도에 더 가까움.
- 앙상블·LLM 결정 모델·sap_rpt, 컨텍스트 탐색 실험(results.csv note 가 `ctx ` 로 시작, 테스트 예측 없음)은 제외. 앙상블 LB 는 [EXPERIMENTS 앙상블 제출](../EXPERIMENTS.md#앙상블-제출).
  `(kNN 1000)` 은 포함: S4E2·Spaceship 에서 kumo·exaone·tabpfn_v2 는 전체 컨텍스트로 실패해 kNN 1000 행 컨텍스트가 유일한 5-fold·제출 결과.
- TFM 컨텍스트 상한: 대회 config 의 `tfm_max_rows` ({capped}) 는 무작위 샘플, 나머지 대회는 상한 없이 학습 fold 전체.
  `(ctx 전체)` / `(ctx 20만)` = 그 대회들에서 상한을 풀거나 20만 행으로 올린 실행 (`@full` / `@rand200000`).

## 종합

평균 CV 순위 = 대회마다 (순위−1)/(모델 수−1) 의 평균 (0 = 항상 1위, 1 = 항상 꼴찌). 실행한 대회가 모델마다 달라 대략적인 비교.
모델마다 대회별로 CV 가 가장 좋은 설정 하나만 셈 (`컨텍스트 변형` 열 = 그중 `@` 설정 수). 최고 LB · s/1K 도 그 행의 값.
s/1K 중앙값: `DBX` = 2026-10-23 이후 Databricks 실행만 (TFM H100, GBDT CPU 4코어), `전체` = 환경 섞임.

**해석 주의**
- **조건이 같지 않을 수 있음**: 상한이 걸린 대회에서 `@full`·`@rand` 로 다시 돌린 모델만 큰 컨텍스트를 씀 (2026-10-26 기준 kumo 뿐).
  다른 TFM 은 상한 그대로라, 그 모델의 CV 1위·평균 순위 일부는 모델 차이가 아니라 컨텍스트 차이
  (예: S6E10 fold 0 에서 tabicl 무작위 20만 행 0.9577, [EXPERIMENTS](../EXPERIMENTS.md#s6e10)).
- **사전학습 데이터 중복 가능성**: Otto·GiveMeSomeCredit·Amazon 등은 OpenML 등에 공개된 데이터라 TFM 사전학습에 학습 데이터가 들어갔을 수 있음 → CV 는 부풀었을 수 있음.
  테스트 라벨은 공개된 적이 없어 LB 점수는 영향 없음. 확인은 안 함.
"""


def render(d: pd.DataFrame, sizes: dict) -> str:
    capped = {}  # 상한 → 대회들
    for c in sorted(d.comp.unique(), key=lambda c: (sizes[c] or (0, 0))[0]):
        if (n := load_competition(c).tfm_max_rows):
            capped.setdefault(n, []).append(TITLES.get(c, c))
    capped = "; ".join(f"{n:,}행: {', '.join(cs)}" for n, cs in capped.items()) or "없음"
    head = HEADER.format(date=pd.Timestamp.now().strftime("%Y-%m-%d"), capped=capped)
    return "\n".join([head, *summary(d), "", "## 대회별\n", *per_comp(d, sizes)])


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km board")
    ap.add_argument("-o", "--out", default=str(OUT), help="출력 경로 (- 면 화면에 출력). 기본: docs/leaderboard.md")
    args = ap.parse_args(argv)
    text = render(*load())
    if args.out == "-":
        print(text)
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"저장: {out}")
