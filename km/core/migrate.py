"""일회성 데이터 마이그레이션. 예: uv run km migrate test-info [--dry-run]

test-info: test.json 이 없는 테스트 예측(test.npy)에 출처를 채움 (test.json 도입 전에 만든 예측).
  출처 = meta.json 의 실행이 results.csv 에서 테스트 예측 실행으로 확인된 것만 (시각·comp·model 일치, made_test).
  확인 못 하면 비워 둠 (추측하지 않음 → km submit 설명에 CV·run 이 안 붙고 km lb --submit 도 자동 제출 안 함).
  이미 test.json 이 있는 모델은 건드리지 않음 → 여러 번 실행해도 같은 결과.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from km.core.results import RESULTS, TEST_INFO, made_test, save_test_info

SOURCE = "backfill: meta.json (2026-10-27 이전 실행)"


def backfill_test_info(dry_run: bool = False) -> tuple[list[str], list[str]]:
    """(채운 모델, 비워 둔 모델과 이유). 모델 이름은 `대회/모델`."""
    res = pd.read_csv(RESULTS / "results.csv")
    res = res.assign(test=made_test(res))
    filled, skipped = [], []
    for d in sorted(RESULTS.glob("*/*/")):
        if not (d / "test.npy").exists() or (d / TEST_INFO).exists():
            continue
        name = f"{d.parent.name}/{d.name}"
        if not (d / "meta.json").exists():
            skipped.append(f"{name}: meta.json 없음")
            continue
        meta = json.loads((d / "meta.json").read_text())
        hit = res[(res.comp == meta["comp"]) & (res.model == meta["model"])
                  & (res.timestamp == meta["timestamp"]) & res.test]
        if not len(hit):
            skipped.append(f"{name}: 실행 {meta['timestamp']} 이 results.csv 에 테스트 예측 실행으로 없음")
            continue
        info = dict(run=meta["timestamp"], cv=meta["scores"][meta["metric"]], n_folds=meta["n_folds"], source=SOURCE)
        if not dry_run:
            save_test_info(d, info)
        filled.append(name)
    return filled, skipped


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km migrate")
    ap.add_argument("name", choices=["test-info"], help="test-info: test.json 없는 예측에 출처(meta.json + results.csv) 채움")
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 결과만 출력")
    args = ap.parse_args(argv)
    filled, skipped = backfill_test_info(args.dry_run)
    verb = "채울 예정" if args.dry_run else "채움"
    print(f"test.json {verb}: {len(filled)}개")
    for s in skipped:
        print(f"  비워 둠 — {s}")
