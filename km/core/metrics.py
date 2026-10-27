"""평가 지표. 분류는 확률 행렬 (n, n_classes), 회귀는 예측값 (n,) 을 받음."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def _accuracy(y, p):
    return skm.accuracy_score(y, p.argmax(1))


def _auc(y, p):
    return skm.roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else skm.roc_auc_score(y, p, multi_class="ovr")


METRICS = {
    # name: (fn, higher_is_better)
    "accuracy": (_accuracy, True),
    "balanced_accuracy": (lambda y, p: skm.balanced_accuracy_score(y, p.argmax(1)), True),
    "auc": (_auc, True),
    "logloss": (lambda y, p: skm.log_loss(y, p, labels=np.arange(p.shape[1])), False),
    "rmse": (lambda y, p: skm.root_mean_squared_error(y, p), False),
    "mae": (lambda y, p: skm.mean_absolute_error(y, p), False),
    "r2": (lambda y, p: skm.r2_score(y, p), True),
}

# 대회 지표 외에 같이 기록할 보조 지표
EXTRA = {"binary": ["accuracy", "auc", "logloss"], "multiclass": ["accuracy", "logloss"], "regression": ["rmse", "mae"]}


def score(name: str, y: np.ndarray, pred: np.ndarray) -> float:
    return float(METRICS[name][0](y, pred))
