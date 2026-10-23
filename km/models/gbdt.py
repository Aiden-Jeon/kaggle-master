"""GBDT 베이스라인. 튜닝 없는 무난한 고정 파라미터 (공정 비교용 기준선)."""

from __future__ import annotations

import os

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


def threads() -> int:
    """CPU 스레드 수. 병렬 작업자(km run 의 run_parallel)는 KM_THREADS 로 CPU 를 나눠 받음, 없으면 -1 = 전체."""
    return int(os.environ.get("KM_THREADS", -1))


def make_lgbm(task, cat_cols, seed):
    import lightgbm as lgb

    params = dict(n_estimators=800, learning_rate=0.02, num_leaves=31, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=threads())
    return lgb.LGBMRegressor(**params) if task == "regression" else lgb.LGBMClassifier(**params)


def make_xgb(task, cat_cols, seed):
    import xgboost as xgb

    params = dict(n_estimators=800, learning_rate=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=threads())
    return xgb.XGBRegressor(**params) if task == "regression" else xgb.XGBClassifier(**params)


class _CatBoost(BaseEstimator, ClassifierMixin):
    """CatBoost 는 category dtype/NaN 범주를 직접 못 받으므로 문자열로 변환."""

    def __init__(self, task, cat_cols, seed):
        self.task, self.cat_cols, self.seed = task, cat_cols, seed

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].astype(object).where(X[c].notna(), "__NA__").astype(str).astype(object)
        return X

    def fit(self, X, y):
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostRegressor if self.task == "regression" else CatBoostClassifier
        self.model_ = cls(iterations=1500, learning_rate=0.03, depth=6, random_seed=self.seed, verbose=0,
                          cat_features=self.cat_cols, allow_writing_files=False, thread_count=threads())
        self.model_.fit(self._prep(X), y)
        return self

    def predict(self, X):
        return self.model_.predict(self._prep(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prep(X))


def make_catboost(task, cat_cols, seed):
    return _CatBoost(task, cat_cols, seed)
