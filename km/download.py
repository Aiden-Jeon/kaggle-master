"""대회 데이터 다운로드: uv run km download spaceship-titanic

사전 준비: Kaggle API 토큰 설정 + 대회 페이지에서 Rules 동의.
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile

from km.competition import Competition, load_competition


def download(comp: Competition):
    comp.data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", comp.name, "-p", str(comp.data_dir)], check=True)
    for z in comp.data_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(comp.data_dir)
        z.unlink()
    print(sorted(p.name for p in comp.data_dir.iterdir()))


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km download")
    ap.add_argument("comp")
    download(load_competition(ap.parse_args(argv).comp))
