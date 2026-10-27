"""제출 노트북 본문 (인터넷 없음). build.py 가 앞에 CODE(km 코드 묶음)·MODELS·FAKE_TEST 를 붙여 kernel 스크립트를 만듦.

  1. TF-IDF 로지스틱 회귀 test 확률 — LB 에서 통과한 탐침(probe_lr.py)과 같은 경로·코드. 메모리를 많이 써서 먼저 끝내고 반환
  2. ntl-prep-offline 출력(kernel_sources)에서 kumo wheel 설치, HF 캐시를 오프라인으로 사용
  3. CV 와 같은 코드로 피처(features.build) → MODELS 의 표 모델을 전체 train 으로 학습해 test 확률 → 1 과 평균
숨겨진 재실행은 로그를 볼 수 없음 → 2·3 이 실패하면 1 만으로 제출 (LB ~0.32 면 이 경로).
"""

import base64
import gc
import io
import os
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 대회 페이지가 안내하는 경로. 숨겨진 재실행에는 sample_submission.parquet 이 없을 수 있음 (그 파일로 찾다가 실패)
COMP_DATA = Path("/kaggle/input/competitions/name-that-llm")
if not (COMP_DATA / "test.csv").exists():
    COMP_DATA = next(p.parent for p in Path("/kaggle/input").rglob("test.csv") if (p.parent / "train.csv").exists())
train, test = pd.read_csv(COMP_DATA / "train.csv"), pd.read_csv(COMP_DATA / "test.csv")
if FAKE_TEST:  # 디버그: 숨겨진 재실행(~4,000행)을 흉내 내 학습 글로 큰 테스트를 만듦
    test = train.sample(FAKE_TEST, replace=True, random_state=0)[["text"]].assign(id=[f"f{i}" for i in range(FAKE_TEST)])
classes = np.array(sorted(train.model.unique()))
y = np.searchsorted(classes, train.model)

vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
vc = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False, min_df=3, sublinear_tf=True,
                     max_features=300_000)
Xs = hstack([vw.fit_transform(train.text), vc.fit_transform(train.text)]).tocsr()
Xs_t = hstack([vw.transform(test.text), vc.transform(test.text)]).tocsr()
p_lr = LogisticRegression(C=10, max_iter=3000).fit(Xs, y).predict_proba(Xs_t)
del vw, vc, Xs, Xs_t  # 글자 n-gram 어휘가 커서 GPU 단계 전에 메모리 반환 (같이 두면 Kaggle 에서 OOM 으로 죽음)
gc.collect()
print("tfidf_lr 완료", flush=True)

probs = [p_lr]
SRC = Path("/kaggle/working/km")
try:
    prep = next(p.parent for p in Path("/kaggle/input").rglob("wheels") if p.parent.name == "offline")
    os.environ.update(HF_HOME=str(prep / "hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", str(prep / "wheels"),
                    "structured-data-models"], check=True)
    tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz").extractall(SRC)
    sys.path.insert(0, str(SRC))
    import torch

    from km.core.competition import load_competition, load_features
    from km.models import make_model

    comp = load_competition("name-that-llm")
    comp.data_dir.mkdir(parents=True, exist_ok=True)  # 임베딩 캐시 위치 (features.cached_embed)
    X, X_test, cat_cols = load_features(comp).build(train, test.reset_index(drop=True))
    print(f"train {X.shape}, test {X_test.shape}", flush=True)
    for name in MODELS:
        model = make_model(name, comp.task, cat_cols, seed=comp.cv["seed"], max_rows=comp.tfm_max_rows,
                           max_features=comp.tfm_max_features, metric=comp.metric)
        p = model.fit(X, y).predict_proba(X_test)
        assert p.shape == p_lr.shape and np.isfinite(p).all(), f"{name} 예측 이상: {p.shape}"
        probs.append(p)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{name} 완료", flush=True)
except Exception:
    traceback.print_exc()
    probs = [p_lr]
    print("피처·표 모델 실패 → 로지스틱 회귀만 사용", flush=True)

pred = classes[np.mean(probs, 0).argmax(1)]
pd.DataFrame({"id": test.id.to_numpy(), "model": pred}).to_parquet("/kaggle/working/submission.parquet", index=False)
print(f"모델 {len(probs)}개 평균\n" + pd.Series(pred).value_counts().to_string())
subprocess.run(["rm", "-rf", str(SRC)])
