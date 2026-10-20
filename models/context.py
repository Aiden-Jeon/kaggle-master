"""컨텍스트 선택 래퍼: 학습 데이터 중 일부만 골라 모델 컨텍스트(학습 데이터)로 넘김.

가설: 쿼리와 비슷한 학습 샘플 k개만으로도 전체 데이터를 넣은 것만큼 추론할 수 있다.

모델 이름 뒤에 `@<spec>` 을 붙여 사용. 예: tabpfn_v2@knn1000, tabicl@knn2000_g64_imp_mix20, tabpfn_v2@rand1000
  mode   full | rand | knn  (full: 학습 fold 전체, rand: 무작위 k행, knn: 쿼리와 가까운 k행)
  k      컨텍스트 행 수 (full 은 생략)
  gN     knn 에서 컨텍스트 하나를 공유하는 쿼리 묶음 크기. 1(기본)이면 샘플마다 따로, 클수록 빠르고 거칠어짐
  imp    거리 공간을 LightGBM gain 중요도로 가중 (기본: 표준화 수치 + 원핫 범주)
  mixP   컨텍스트의 P% 는 무작위 행으로 채움 (이웃이 한 클래스로 쏠리는 것 완화)

분류에서는 컨텍스트에 빠진 클래스가 있으면 클래스마다 무작위 2행을 추가해 모든 클래스가 들어가게 함.
"""

from __future__ import annotations

import math
import re
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

_SPEC = re.compile(r"^(full|rand|knn)(\d+)?((?:_[a-z]+\d*)*)$")
_MIN_PER_CLASS = 2
_MAX_LEVELS = 20  # 원핫으로 펼칠 범주 수 상한 (나머지는 0 벡터)


def parse_spec(spec: str) -> dict:
    m = _SPEC.match(spec)
    if not m:
        raise ValueError(f"컨텍스트 spec '{spec}' 형식 오류. 예: knn1000, rand2000, knn1000_g64_imp_mix20, full")
    mode, k, opts = m.group(1), m.group(2), m.group(3)
    if (mode == "full") != (k is None):
        raise ValueError(f"'{spec}': full 은 k 없이, rand/knn 은 k 와 함께 지정")
    out = dict(mode=mode, k=int(k) if k else None, group=1, space="raw", mix=0.0)
    for opt in filter(None, opts.split("_")):
        if opt == "imp":
            out["space"] = "imp"
        elif opt.startswith("g") and opt[1:].isdigit():
            out["group"] = int(opt[1:])
        elif opt.startswith("mix") and opt[3:].isdigit():
            out["mix"] = int(opt[3:]) / 100
        else:
            raise ValueError(f"'{spec}': 알 수 없는 옵션 '{opt}'")
    return out


class _Encoder:
    """kNN 거리 공간: 수치형은 중앙값 대체 후 표준화, 범주형은 원핫 × 1/√2 (불일치 시 거리² 1)."""

    def __init__(self, cat_cols, task, space, seed):
        self.cat_cols, self.task, self.space, self.seed = cat_cols, task, space, seed

    def fit(self, X: pd.DataFrame, y):
        self.num_ = [c for c in X.columns if c not in self.cat_cols]
        self.med_ = X[self.num_].median()
        self.sd_ = X[self.num_].fillna(self.med_).std().replace(0, 1).fillna(1)
        self.levels_ = {c: X[c].value_counts().index[:_MAX_LEVELS] for c in self.cat_cols}
        owner = self.num_ + [c for c in self.cat_cols for _ in self.levels_[c]]  # 출력 열마다 원래 피처
        self.scale_ = np.ones(len(owner), dtype=np.float32)
        if self.space == "imp":
            imp = self._importance(X, y)
            self.scale_ = np.sqrt([imp[c] for c in owner]).astype(np.float32)
        return self

    def _importance(self, X, y) -> pd.Series:
        import lightgbm as lgb

        cls = lgb.LGBMRegressor if self.task == "regression" else lgb.LGBMClassifier
        if len(X) > 50_000:
            idx = np.random.default_rng(self.seed).choice(len(X), 50_000, replace=False)
            X, y = X.iloc[idx], np.asarray(y)[idx]
        model = cls(n_estimators=200, importance_type="gain", random_state=self.seed, verbose=-1).fit(X, y)
        imp = pd.Series(model.feature_importances_, index=X.columns).clip(lower=0)
        return imp / imp.mean() if imp.sum() > 0 else imp * 0 + 1  # 평균 1 로 정규화

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts = [((X[self.num_].fillna(self.med_) - self.med_) / self.sd_).to_numpy(np.float32)]
        for c in self.cat_cols:
            parts.append((X[c].to_numpy()[:, None] == np.asarray(self.levels_[c])[None]).astype(np.float32)
                         / math.sqrt(2))
        return np.hstack(parts) * self.scale_


class ContextSelect(BaseEstimator):
    def __init__(self, make_base, task, cat_cols, mode, k=None, group=1, space="raw", mix=0.0, seed=0):
        self.make_base, self.task, self.cat_cols = make_base, task, cat_cols
        self.mode, self.k, self.group, self.space, self.mix, self.seed = mode, k, group, space, mix, seed

    @property
    def _is_clf(self):
        return self.task != "regression"

    def fit(self, X, y):
        self.X_, self.y_ = X.reset_index(drop=True), np.asarray(y)
        self.rng_ = np.random.default_rng(self.seed)
        if self._is_clf:
            self.classes_ = np.unique(self.y_)
        if self.mode == "knn":
            self.enc_ = _Encoder(self.cat_cols, self.task, self.space, self.seed).fit(self.X_, self.y_)
            self._index(self.enc_.transform(self.X_))
            self.base_ = None
        else:
            idx = np.arange(len(self.X_)) if self.mode == "full" else self._random(self.k)
            self.base_ = self._fit_base(idx)
        return self

    # --- 컨텍스트 구성 ---

    def _random(self, n, exclude=None):
        pool = np.arange(len(self.X_)) if exclude is None else np.setdiff1d(np.arange(len(self.X_)), exclude)
        return self._cover_classes(self.rng_.choice(pool, min(n, len(pool)), replace=False))

    def _cover_classes(self, idx):
        if not self._is_clf:
            return idx
        missing = np.setdiff1d(self.classes_, self.y_[idx])
        extra = [self.rng_.choice(np.flatnonzero(self.y_ == c), _MIN_PER_CLASS) for c in missing]
        return np.concatenate([idx, *extra]) if extra else idx

    def _knn_context(self, Q):
        """쿼리 묶음 Q 의 컨텍스트: 쿼리별 이웃 순위를 번갈아(1순위들 → 2순위들 → …) 모아 중복 제거."""
        n_rand = round(self.k * self.mix)
        n_nn = self.k - n_rand
        nb = self._topk(Q, n_nn)
        order = nb.T.ravel()
        _, first = np.unique(order, return_index=True)
        idx = order[np.sort(first)][:n_nn]
        if n_rand:
            idx = np.concatenate([idx, self.rng_.choice(np.setdiff1d(np.arange(len(self.X_)), idx), n_rand,
                                                        replace=False)])
        return self._cover_classes(idx)

    # --- 최근접 이웃 (GPU 가 있으면 torch, 없으면 numpy) ---
    # CPU 에서 torch 를 쓰지 않는 이유: macOS 에서 LightGBM(imp) 실행 후 CPU torch 연산이 OpenMP 충돌로 멈춤

    def _index(self, Z):
        import torch

        self.Z_, self.z2_ = Z, (Z ** 2).sum(1)
        self.gpu_ = torch.cuda.is_available()
        if self.gpu_:
            self.Zt_, self.z2t_ = torch.from_numpy(Z).cuda(), torch.from_numpy(self.z2_).cuda()

    def _topk(self, Q, k, chunk=256):
        k = min(k, len(self.z2_))
        out = []
        for s in range(0, len(Q), chunk):
            if self.gpu_:
                import torch

                d = self.z2t_[None] - 2 * torch.from_numpy(Q[s:s + chunk]).cuda() @ self.Zt_.T  # |q|² 는 순위 무관
                out.append(d.topk(k, dim=1, largest=False).indices.cpu().numpy())
            else:
                d = self.z2_[None] - 2 * Q[s:s + chunk] @ self.Z_.T
                part = np.argpartition(d, k - 1, axis=1)[:, :k]
                out.append(np.take_along_axis(part, np.take_along_axis(d, part, 1).argsort(1), 1))
        return np.vstack(out)

    def _groups(self, Q):
        if self.group <= 1:
            return [np.array([i]) for i in range(len(Q))]
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = max(1, math.ceil(len(Q) / self.group))
        labels = MiniBatchKMeans(n_clusters, random_state=self.seed, n_init=1, batch_size=4096).fit_predict(Q)
        groups = []
        for c in np.unique(labels):  # 너무 큰 클러스터는 묶음 크기에 맞게 쪼갬
            members = np.flatnonzero(labels == c)
            groups += np.array_split(members, max(1, round(len(members) / self.group)))
        return groups

    # --- 학습/예측 ---

    def _fit_base(self, idx):
        return self.make_base().fit(self.X_.iloc[idx], self.y_[idx])

    def _base_out(self, base, X, ctx_y):
        if not self._is_clf:
            return base.predict(X)
        p = np.zeros((len(X), len(self.classes_)))
        p[:, np.searchsorted(self.classes_, np.unique(ctx_y))] = base.predict_proba(X)
        return p

    def _run(self, X):
        X = X.reset_index(drop=True)
        if self.base_ is not None:
            return self._base_out(self.base_, X, self.y_)
        t0 = time.time()
        Q = self.enc_.transform(X)
        groups = self._groups(Q)
        out = np.zeros((len(X), len(self.classes_))) if self._is_clf else np.zeros(len(X))
        purity, sizes = [], []
        for i, g in enumerate(groups):
            idx = self._knn_context(Q[g])
            out[g] = self._base_out(self._fit_base(idx), X.iloc[g], self.y_[idx])
            sizes.append(len(idx))
            if self._is_clf:
                purity.append(np.bincount(self.y_[idx]).max() / len(idx))
            if (i + 1) % max(1, len(groups) // 10) == 0:
                print(f"    [ctx] {i + 1}/{len(groups)} 묶음 ({time.time() - t0:.0f}s)", flush=True)
        msg = f"    [ctx] knn k={self.k} 쿼리 {len(X)} → 묶음 {len(groups)}, 평균 컨텍스트 {np.mean(sizes):.0f}행"
        if purity:
            msg += f", 다수 클래스 비율 {np.mean(purity):.3f}"
        print(msg + f" ({time.time() - t0:.0f}s)", flush=True)
        return out

    def predict(self, X):
        out = self._run(X)
        return self.classes_[out.argmax(1)] if self._is_clf else out

    def predict_proba(self, X):
        return self._run(X)


def wrap(make_base, spec, task, cat_cols, seed):
    return ContextSelect(make_base, task, cat_cols, seed=seed, **parse_spec(spec))
