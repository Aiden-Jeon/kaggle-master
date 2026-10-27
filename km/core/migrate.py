"""일회성 데이터 마이그레이션. 예: uv run km migrate test-info [--dry-run]

test-info: test.json 이 없는 테스트 예측(test.npy)에 출처를 채움 (test.json 도입 전에 만든 예측).
  출처 = meta.json 의 실행이 results.csv 에서 테스트 예측 실행으로 확인된 것만 (시각·comp·model 일치, made_test).
  확인 못 하면 비워 둠 (추측하지 않음 → km submit 설명에 CV·run 이 안 붙고 km lb --submit 도 자동 제출 안 함).
  이미 test.json 이 있는 모델은 건드리지 않음 (이 명령이 채운 것에 classes 만 없으면 채움) → 여러 번 실행해도 같은 결과.
"""

from __future__ import annotations

import argparse
import json

from km.core.results import RESULTS, TEST_INFO, made_test, parse_utc, read_results, save_test_info, test_info

SOURCE = "backfill: meta.json (2026-10-27 이전 실행)"


def run_time(text):
    """parse_utc, 못 읽으면 None (그 행·파일은 어느 실행과도 맞지 않는 것으로 봄 → 마이그레이션 전체를 멈추지 않음)."""
    try:
        return parse_utc(text)
    except RuntimeError:
        return None


def backfill_test_info(dry_run: bool = False) -> tuple[list[str], list[str]]:
    """(채운 모델, 비워 둔 모델과 이유). 모델 이름은 `대회/모델`.
    이 명령으로 채운 test.json (source = SOURCE) 에 classes 가 없으면 (2026-10-27 일회성 백필) 같은 실행의 meta.json 에서 채움."""
    res = read_results()
    res = res.assign(test=made_test(res), run_at=res.timestamp.map(run_time))
    filled, skipped = [], []
    for d in sorted(RESULTS.glob("*/*/")):
        if not (d / "test.npy").exists():
            continue
        name = f"{d.parent.name}/{d.name}"
        try:
            meta = json.loads((d / "meta.json").read_text())
            run, comp, model = meta["timestamp"], meta["comp"], meta["model"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            if not (d / TEST_INFO).exists():
                skipped.append(f"{name}: meta.json 을 읽지 못함 ({e!r})")
            continue
        if (d / TEST_INFO).exists():
            info = test_info(d.parent.name, d.name)
            # 예측 열 순서 (classes) 가 없으면 km submit 이 meta.json 으로 대신함 → --no-test 재실행이 덮어쓰면 틀림
            if (info and info.get("source") == SOURCE and "classes" not in info
                    and run_time(run) is not None and run_time(info["run"]) == run_time(run)):
                if not dry_run:
                    save_test_info(d, {**info, "classes": meta.get("classes")})
                filled.append(f"{name} (classes)")
            continue
        at = run_time(run)
        hit = res[(res.comp == comp) & (res.model == model) & (res.run_at == at) & res.test] if at else res.iloc[:0]
        if not len(hit):
            skipped.append(f"{name}: 실행 {run} 이 results.csv 에 테스트 예측 실행으로 없음")
            continue
        try:
            info = dict(run=run, cv=meta["scores"][meta["metric"]], n_folds=meta["n_folds"], classes=meta.get("classes"),
                        source=SOURCE)
        except (KeyError, TypeError) as e:
            skipped.append(f"{name}: meta.json 에 CV·fold 수가 없음 ({e!r})")
            continue
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
