"""LB 탐침: TF-IDF 로지스틱 회귀 (+ 주제 묶음 배정). 외부 패키지·인터넷 없이 Kaggle 기본 이미지에서 돎.

ASSIGN = False: 확률 최고 클래스 / True: 테스트를 주제(TF-IDF 코사인) 로 묶고 묶음 안에서 모델당 최대 1번 배정
(prompt 하나에 모델마다 한 번씩 답했다는 학습 데이터 구조를 테스트에도 가정). 로컬 확인: DATA·OUT 환경변수.
2026-10-27 public LB: False 0.322 (CV 0.322) / True 0.321 → 숨겨진 테스트에서는 배정 이득 없음 (submit.py 에서 뺌).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import hstack
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ASSIGN = False
THRESHOLD = 0.9  # 클러스터 거리 임계값 (코사인). 모델별 글 수가 다른 시뮬레이션에서 0.85~0.95 중 최고
DATA = Path(os.environ.get("DATA", "/kaggle/input/competitions/name-that-llm"))
OUT = Path(os.environ.get("OUT", "."))

train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
classes = np.array(sorted(train.model.unique()))
y = np.searchsorted(classes, train.model)

vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
vc = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False, min_df=3, sublinear_tf=True,
                     max_features=300_000)
X = hstack([vw.fit_transform(train.text), vc.fit_transform(train.text)]).tocsr()
Xt = hstack([vw.transform(test.text), vc.transform(test.text)]).tocsr()
P = LogisticRegression(C=10, max_iter=3000).fit(X, y).predict_proba(Xt)
pred = P.argmax(1)

if ASSIGN and len(test) > 1:
    T = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(test.text).toarray()
    # 다른 글과 겹치는 단어가 없는 글(영벡터)은 코사인을 못 구함 → 혼자 한 묶음
    nz = T.any(1)
    groups = -1 - np.arange(len(test))
    if nz.sum() > 1:
        groups[nz] = AgglomerativeClustering(n_clusters=None, distance_threshold=THRESHOLD, metric="cosine",
                                             linkage="average").fit_predict(T[nz])
    K = len(classes)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        cap = -(-len(idx) // K)  # 묶음이 모델 수보다 크면(클러스터가 prompt 여러 개를 합침) 모델당 여러 번 허용
        r, c = linear_sum_assignment(-np.log(np.repeat(P[idx], cap, axis=1) + 1e-9))
        pred[idx[r]] = c // cap
    print(f"묶음 {len(np.unique(groups))}개, 확률 최고 클래스와 다른 예측 {(pred != P.argmax(1)).mean():.1%}")

pd.DataFrame({"id": test.id, "model": classes[pred]}).to_parquet(OUT / "submission.parquet", index=False)
print(pd.Series(classes[pred]).value_counts().to_string())
