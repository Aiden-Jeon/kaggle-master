"""대회 데이터 다운로드: uv run km download spaceship-titanic

사전 준비: Kaggle API 토큰 설정 + 대회 페이지에서 Rules 동의.
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

from km.competition import Competition, load_competition


def extract(data_dir: Path):
    """data_dir 의 zip 을 (zip 안의 zip 까지) 풀고 zip 은 지움. Kaggle·Colab 실행 코드도 이 함수를 씀.

    이미 있는 파일은 덮어쓰지 않고, 같은 zip 안에 x 와 x.zip 이 같이 있으면 x.zip 은 건너뜀
    (옛 대회 번들은 x.csv 와 x.csv.zip 을 같이 담기도 함. allstate: zip 쪽은 깨진 옛 버전, Mercedes: zip 만 있음).
    zip 이 심볼릭 링크(Kaggle 의 읽기 전용 입력)여도 링크만 지움.
    """
    while zips := list(data_dir.glob("*.zip")):
        for z in zips:
            with zipfile.ZipFile(z) as f:
                names = set(f.namelist())
                f.extractall(data_dir, members=[n for n in names if not (data_dir / n).exists()
                                                and (not n.endswith(".zip") or n.removesuffix(".zip") not in names)])
            z.unlink()


def download(comp: Competition):
    comp.data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", comp.name, "-p", str(comp.data_dir)], check=True)
    extract(comp.data_dir)
    print(sorted(p.name for p in comp.data_dir.iterdir()))


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km download")
    ap.add_argument("comp")
    download(load_competition(ap.parse_args(argv).comp))
