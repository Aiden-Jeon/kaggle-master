"""Mercedes 피처: 범주형 X0~X8 (문자열) + 0/1 플래그 ~370개. 학습에서 값이 하나뿐인 열은 버림."""

from __future__ import annotations

import pandas as pd

from km.features import auto_types, split


def build(train: pd.DataFrame, test: pd.DataFrame):
    const = [c for c in train.columns if train[c].nunique() <= 1]
    df = pd.concat([train, test], keys=["train", "test"]).drop(columns=["ID", "y", *const], errors="ignore")
    df, cat_cols = auto_types(df)
    return *split(df), cat_cols
