"""제출 kernel 폴더 만들기: python competitions/name-that-llm/kernel/build.py <출력 폴더> <kernel id> [모델...]

submit.py 앞에 km 코드 묶음과 설정을 붙이고 kernel-metadata.json (GPU, 인터넷 끔, ntl-prep-offline 출력 연결) 작성.
이후 `kaggle kernels push -p <출력 폴더>`.
"""

import argparse
import json
from pathlib import Path

from km.runner.common import bundle

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("kernel")
ap.add_argument("models", nargs="*", default=["kumo"])
ap.add_argument("--fake-test", type=int, default=0, help="디버그: 학습 글로 만든 n행 가짜 테스트로 실행")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
head = f"CODE = {bundle()!r}\nMODELS = {args.models!r}\nFAKE_TEST = {args.fake_test!r}\n\n"
(out / "submit.py").write_text(head + (Path(__file__).parent / "submit.py").read_text())
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": args.kernel, "title": args.kernel.split("/")[1], "code_file": "submit.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False,
    "competition_sources": ["name-that-llm"], "dataset_sources": [],
    "kernel_sources": [f"{args.kernel.split('/')[0]}/ntl-prep-offline"],
}, indent=2))
print(f"{out}: models={args.models} fake_test={args.fake_test}")
