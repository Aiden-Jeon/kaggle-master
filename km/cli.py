"""`km <명령> ...` 진입점. 명령마다 해당 모듈의 main(argv) 로 넘김 (필요한 모듈만 import)."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    # 이름: (모듈, 설명)
    "run": ("km.runner.run", "고정 CV fold 로 모델 실행 → OOF/테스트 예측, results.csv"),
    "submit": ("km.leaderboard.submit", "저장된 테스트 예측으로 제출 파일 생성 (--push 로 Kaggle 제출)"),
    "download": ("km.core.download", "대회 데이터 다운로드"),
    "lb": ("km.leaderboard.rank", "Kaggle 제출 기록 → 리더보드 순위 (results/leaderboard.csv)"),
    "board": ("km.leaderboard.board", "results.csv·leaderboard.csv → 모델별 성능·실행 시간 리더보드 (docs/leaderboard.md, .html)"),
    "kaggle": ("km.runner.kaggle", "Kaggle Notebook(GPU)에서 km run 실행 후 결과 수집"),
    "colab": ("km.runner.colab", "Colab 노트북 생성 / 출력에서 결과 수집"),
    "databricks": ("km.runner.databricks", "Databricks AI Runtime(서버리스 GPU)에서 km run 실행 후 결과 수집"),
    "migrate": ("km.core.migrate", "일회성 데이터 마이그레이션 (km migrate test-info: test.json 없는 예측에 출처 채움)"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"  {name:<{width}}  {desc}" for name, (_, desc) in COMMANDS.items()]
    return "사용: km <명령> [인자...]   (명령별 도움말: km <명령> -h)\n\n" + "\n".join(lines)


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return
    if argv[0] not in COMMANDS:
        sys.exit(f"알 수 없는 명령 '{argv[0]}'\n\n{usage()}")
    importlib.import_module(COMMANDS[argv[0]][0]).main(argv[1:])
