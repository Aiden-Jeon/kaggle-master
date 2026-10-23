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
    while zips := list(comp.data_dir.glob("*.zip")):  # 번들 안에 든 zip 까지 다 풀 때까지
        for z in zips:
            with zipfile.ZipFile(z) as f:
                names = set(f.namelist())
                # 옛 대회 번들은 x.csv 와 x.csv.zip 을 같이 담기도 함 (allstate: zip 쪽은 깨진 옛 버전) → x.csv 만
                f.extractall(comp.data_dir, members=[n for n in names if n.removesuffix(".zip") == n
                                                     or n.removesuffix(".zip") not in names])
            z.unlink()
    print(sorted(p.name for p in comp.data_dir.iterdir()))


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="km download")
    ap.add_argument("comp")
    download(load_competition(ap.parse_args(argv).comp))
