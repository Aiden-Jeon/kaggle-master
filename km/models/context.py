"""컨텍스트 선택 래퍼: 학습 데이터 중 일부만 골라 모델 컨텍스트(학습 데이터)로 넘김.

가설: 쿼리와 비슷한 학습 샘플 k개만으로도 전체 데이터를 넣은 것만큼 추론할 수 있다.

모델 이름 뒤에 `@<spec>` 을 붙여 사용. 예: tabpfn_v2@knn1000, tabicl@knn2000_g64_imp_mix20, tabpfn_v2@rand1000
  mode   full | rand | knn | part
         full: 학습 fold 전체, rand: 무작위 k행, knn: 쿼리와 가까운 k행,
         part: 학습 데이터를 약 N/k 개 영역(k-means)으로 나누고 영역 중심에 가까운 k행을 영역 컨텍스트로 사용.
               쿼리는 가장 가까운 영역으로 보냄 → fit 횟수가 쿼리 수가 아니라 영역 수에 비례
  k      컨텍스트 행 수 (full 은 생략)
  gN     knn 에서 컨텍스트 하나를 공유하는 쿼리 묶음 크기. 1(기본)이면 샘플마다 따로, 클수록 빠르고 거칠어짐
  imp    거리 공간을 LightGBM gain 중요도로 가중 (기본: 표준화 수치 + 원핫 범주)
  leaf   거리 공간을 LightGBM leaf 임베딩으로 (같은 leaf 에 떨어지는 트리가 많을수록 가까움)
  mixP   컨텍스트의 P% 는 무작위 행으로 채움 (이웃이 한 클래스로 쏠리는 것 완화)
  balP   컨텍스트의 P% 는 클래스 균형 고정 샘플(코어셋)로 채움 (모든 컨텍스트가 공유)

분류에서는 컨텍스트에 빠진 클래스가 있으면 클래스마다 무작위 2행을 추가해 모든 클래스가 들어가게 함.
"""

from __future__ import annotations

import math
import re
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

_SPEC = re.compile(r"^(full|rand|knn|part)(\d+)?((?:_[a-z]+\d*)*)$")
_LEAF_TREES = 100  # leaf 임베딩에 쓰는 트리 수 (다중 분류는 반복당 클래스 수만큼 트리가 생김)
_MIN_PER_CLASS = 2
_MAX_LEVELS = 20  # 원핫으로 펼칠 범주 수 상한 (나머지는 0 벡터)


def parse_spec(spec: str) -> dict:
    m = _SPEC.match(spec)
    if not m:
        raise ValueError(f"컨텍스트 spec '{spec}' 형식 오류. 예: knn1000, rand2000, knn1000_g64_imp_mix20, full")
    mode, k, opts = m.group(1), m.group(2), m.group(3)
    if (mode == "full") != (k is None):
        raise ValueError(f"'{spec}': full 은 k 없이, rand/knn 은 k 와 함께 지정")
    out = dict(mode=mode, k=int(k) if k else None, group=1, space="raw", mix=0.0, bal=0.0)
    for opt in filter(None, opts.split("_")):
        if opt in ("imp", "leaf"):
            out["space"] = opt
        elif opt.startswith("g") and opt[1:].isdigit():
            out["group"] = int(opt[1:])
        elif opt.startswith("mix") and opt[3:].isdigit():
            out["mix"] = int(opt[3:]) / 100
        elif opt.startswith("bal") and opt[3:].isdigit():
            out["bal"] = int(opt[3:]) / 100
        else:
            raise ValueError(f"'{spec}': 알 수 없는 옵션 '{opt}'")
    return out


def fit_lgbm(X: pd.DataFrame, y, task: str, seed: int, **params):
    """거리 공간·피처 선택용 보조 LightGBM (5만 행 넘으면 무작위 샘플로 학습)."""
    import lightgbm as lgb

    cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    if len(X) > 50_000:
        idx = np.random.default_rng(seed).choice(len(X), 50_000, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    return cls(random_state=seed, verbose=-1, **params).fit(X, y)


def gain_importance(X: pd.DataFrame, y, task: str, seed: int) -> pd.Series:
    """LightGBM gain 중요도, 평균 1 로 정규화."""
    model = fit_lgbm(X, y, task, seed, n_estimators=200, importance_type="gain")
    imp = pd.Series(model.feature_importances_, index=X.columns).clip(lower=0)
    return imp / imp.mean() if imp.sum() > 0 else imp * 0 + 1


class _Encoder:
    """kNN 거리 공간: 수치형은 중앙값 대체 후 표준화, 범주형은 원핫 × 1/√2 (불일치 시 거리² 1).

    space=leaf 이면 대신 LightGBM 의 leaf 원핫 (트리마다 1/√T) → 거리² = 2 × (서로 다른 leaf 에 떨어진 트리 비율).
    """

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
            imp = gain_importance(X, y, self.task, self.seed)
            self.scale_ = np.sqrt([imp[c] for c in owner]).astype(np.float32)
        elif self.space == "leaf":
            per_iter = len(np.unique(y)) if self.task == "multiclass" else 1
            self.gbdt_ = fit_lgbm(X, y, self.task, self.seed, n_estimators=max(10, _LEAF_TREES // per_iter))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.space == "leaf":
            leaves = self.gbdt_.predict(X, pred_leaf=True)  # (n, 트리 수)
            n_trees, n_leaves = leaves.shape[1], self.gbdt_.get_params()["num_leaves"]
            out = np.zeros((len(X), n_trees * n_leaves), dtype=np.float32)
            out[np.arange(len(X))[:, None], leaves + np.arange(n_trees) * n_leaves] = 1 / math.sqrt(n_trees)
            return out
        parts = [((X[self.num_].fillna(self.med_) - self.med_) / self.sd_).to_numpy(np.float32)]
        for c in self.cat_cols:
            parts.append((X[c].to_numpy()[:, None] == np.asarray(self.levels_[c])[None]).astype(np.float32)
                         / math.sqrt(2))
        return np.hstack(parts) * self.scale_


class ContextSelect(BaseEstimator):
    def __init__(self, make_base, task, cat_cols, mode, k=None, group=1, space="raw", mix=0.0, bal=0.0, seed=0):
        self.make_base, self.task, self.cat_cols = make_base, task, cat_cols
        self.mode, self.k, self.group, self.space, self.seed = mode, k, group, space, seed
        self.mix, self.bal = mix, bal

    @property
    def joint_predict(self):
        """검증·테스트를 합쳐 예측해도 되는지 (km run). knn/part 는 쿼리 묶음·난수 순서가 쿼리 집합에 따라 달라져 따로 예측."""
        return self.mode in ("full", "rand")

    @property
    def _is_clf(self):
        return self.task != "regression"

    def fit(self, X, y):
        self.X_, self.y_ = X.reset_index(drop=True), np.asarray(y)
        self.rng_ = np.random.default_rng(self.seed)
        if self._is_clf:
            self.classes_ = np.unique(self.y_)
        if self.mode in ("knn", "part"):
            self.enc_ = _Encoder(self.cat_cols, self.task, self.space, self.seed).fit(self.X_, self.y_)
            Z = self.enc_.transform(self.X_)
            self._index(Z)
            self.core_ = self._balanced(round(self.k * self.bal)) if self.bal else np.array([], dtype=int)
            if self.mode == "part":
                from sklearn.cluster import MiniBatchKMeans

                n_regions = max(1, math.ceil(len(Z) / self.k))
                self.centers_ = MiniBatchKMeans(n_regions, random_state=self.seed, n_init=1,
                                                batch_size=4096).fit(Z).cluster_centers_.astype(np.float32)
            self.base_ = None
        else:
            idx = np.arange(len(self.X_)) if self.mode == "full" else self._random(self.k)
            self.base_ = self._fit_base(idx)
        return self

    # --- 컨텍스트 구성 ---

    def _random(self, n, exclude=None):
        pool = np.arange(len(self.X_)) if exclude is None else np.setdiff1d(np.arange(len(self.X_)), exclude)
        return self._cover_classes(self.rng_.choice(pool, min(n, len(pool)), replace=False))

    def _balanced(self, n):
        """클래스마다 같은 수(클래스 크기 한도 내)로 뽑은 고정 샘플. 회귀는 무작위."""
        if not self._is_clf:
            return self.rng_.choice(len(self.X_), min(n, len(self.X_)), replace=False)
        per = max(1, n // len(self.classes_))
        return np.concatenate([self.rng_.choice(np.flatnonzero(self.y_ == c), min(per, (self.y_ == c).sum()),
                                                replace=False) for c in self.classes_])

    def _cover_classes(self, idx):
        if not self._is_clf:
            return idx
        missing = np.setdiff1d(self.classes_, self.y_[idx])
        extra = [self.rng_.choice(np.flatnonzero(self.y_ == c), _MIN_PER_CLASS) for c in missing]
        return np.concatenate([idx, *extra]) if extra else idx

    def _knn_context(self, Q):
        """기준점 묶음 Q(쿼리들 또는 영역 중심)의 컨텍스트: 점별 이웃 순위를 번갈아(1순위들 → 2순위들 → …)
        모아 중복 제거. bal 이면 코어셋을 먼저 넣고 나머지를 이웃으로, mix 면 일부를 무작위로 채움."""
        n_rand = round(self.k * self.mix)
        n_nn = self.k - n_rand - len(self.core_)
        nb = self._topk(Q, n_nn + len(self.core_))
        order = nb.T.ravel()
        order = order[~np.isin(order, self.core_)]
        _, first = np.unique(order, return_index=True)
        idx = np.concatenate([self.core_, order[np.sort(first)][:n_nn]]).astype(int)
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
        if self.mode == "part":  # 쿼리를 가장 가까운 영역으로, 컨텍스트 기준점은 영역 중심
            region = (((self.centers_ ** 2).sum(1)[None] - 2 * Q @ self.centers_.T)).argmin(1)
            regions = np.unique(region)
            groups = [np.flatnonzero(region == r) for r in regions]
            anchors = [self.centers_[[r]] for r in regions]
        else:
            groups = self._groups(Q)
            anchors = [Q[g] for g in groups]
        out = np.zeros((len(X), len(self.classes_))) if self._is_clf else np.zeros(len(X))
        purity, sizes = [], []
        for i, g in enumerate(groups):
            idx = self._knn_context(anchors[i])
            out[g] = self._base_out(self._fit_base(idx), X.iloc[g], self.y_[idx])
            sizes.append(len(idx))
            if self._is_clf:
                purity.append(np.bincount(self.y_[idx]).max() / len(idx))
            if (i + 1) % max(1, len(groups) // 10) == 0:
                print(f"    [ctx] {i + 1}/{len(groups)} 묶음 ({time.time() - t0:.0f}s)", flush=True)
        msg = f"    [ctx] {self.mode} k={self.k} 쿼리 {len(X)} → 묶음 {len(groups)}, 평균 컨텍스트 {np.mean(sizes):.0f}행"
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
