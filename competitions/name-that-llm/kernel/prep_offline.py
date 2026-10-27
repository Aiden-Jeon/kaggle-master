"""오프라인 제출용 묶음 만들기 (인터넷 켠 Kaggle 노트북에서 실행, 출력을 Dataset 으로 올림).

/kaggle/working/offline/
  wheels/   kumo(structured-data-models) 와 Kaggle 이미지에 없는 의존성 wheel → 제출 노트북에서 --no-index 설치
  hf/       HF 캐시: 임베딩 LM + kumo 가중치 (HF_HOME 으로 지정해 오프라인 로드).
            노트북 출력은 심볼릭 링크를 보존하지 않음 → /tmp 에 받은 뒤 snapshots 의 링크를 실제 파일로 풀어 복사 (blobs 제외)
  cache/    그 밖의 캐시 (XDG_CACHE_HOME·TORCH_HOME). kumo 가 HF 가 아닌 곳에 받는 경우 대비
Kaggle 이미지와 같은 파이썬·torch 에서 받아야 wheel 이 맞으므로 로컬이 아니라 노트북에서 받음.
"""

import os
import subprocess
import sys
from pathlib import Path

OUT = Path("/kaggle/working/offline")
EMB_MODEL = "Qwen/Qwen3-0.6B"
(OUT / "wheels").mkdir(parents=True, exist_ok=True)
HF_TMP = Path("/tmp/hf")
os.environ["HF_HOME"] = str(HF_TMP)
os.environ["XDG_CACHE_HOME"] = os.environ["TORCH_HOME"] = str(OUT / "cache")

pip = [sys.executable, "-m", "pip"]
T0 = Path("/kaggle/working/.t0")
T0.touch()


def freeze() -> set[str]:
    return set(subprocess.run([*pip, "freeze"], capture_output=True, text=True).stdout.split())


# 이미지에 없는 패키지만 wheel 로 받음 (torch 등 이미지에 있는 건 제외 → 묶음이 작음)
before = freeze()
subprocess.run([*pip, "install", "-q", "structured-data-models"], check=True)
new = sorted(p for p in freeze() - before if "==" in p)
print("새로 설치:", new)
subprocess.run([*pip, "download", "-q", "--no-deps", "-d", str(OUT / "wheels"), *new], check=True)

from huggingface_hub import snapshot_download  # noqa: E402

snapshot_download(EMB_MODEL)

# kumo 가중치: 작은 데이터로 한 번 돌려 HF 캐시에 받게 함
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sdm  # noqa: E402
import torch  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.DataFrame({"f0": np.random.rand(40), "target": np.arange(40) % 3})
t = sdm.TableTensor.from_pandas(df, stypes={"f0": "numerical", "target": "categorical"}, device=dev)
m = sdm.models.KumoTabular(task="classification", device=dev)
m.fit(t.drop_columns("target"), t[:, "target"], num_estimators=1)
print(m.predict(t.drop_columns("target")))

import shutil  # noqa: E402

shutil.copytree(HF_TMP / "hub", OUT / "hf" / "hub", symlinks=False, ignore=shutil.ignore_patterns("blobs", ".locks"))
subprocess.run(["du", "-sh", *map(str, OUT.iterdir())])
subprocess.run(["find", str(OUT / "hf"), "-maxdepth", "5"])
# 묶음 밖에 받힌 큰 파일이 있는지 (있으면 제출 노트북에서 못 찾음)
subprocess.run("find / -xdev -size +50M -newer /kaggle/working/.t0 2>/dev/null | grep -v '^/kaggle/working/offline' | grep -v '^/proc'", shell=True)
subprocess.run(["ls", str(OUT / "wheels")])
subprocess.run([*pip, "list"], stdout=open("/kaggle/working/pip_list.txt", "w"))
