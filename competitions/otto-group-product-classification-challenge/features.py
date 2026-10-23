"""Otto 상품 분류 피처: 익명 카운트 피처 93개 그대로."""

from __future__ import annotations

import pandas as pd

from km.features import auto_types, split


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train, test], keys=["train", "test"]).drop(columns=["id", "target"], errors="ignore")
    df, cat_cols = auto_types(df)
    return *split(df), cat_cols
