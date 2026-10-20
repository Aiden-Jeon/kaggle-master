"""대회 데이터 다운로드: uv run python download.py spaceship-titanic

사전 준비: Kaggle API 토큰 설정 + 대회 페이지에서 Rules 동의.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile

from core.competition import load_competition


def main():
    comp = load_competition(sys.argv[1])
    comp.data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", comp.name, "-p", str(comp.data_dir)], check=True)
    for z in comp.data_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(comp.data_dir)
        z.unlink()
    print(sorted(p.name for p in comp.data_dir.iterdir()))


if __name__ == "__main__":
    main()
