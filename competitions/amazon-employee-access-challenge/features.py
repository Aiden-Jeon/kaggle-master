"""Amazon 직원 권한 피처: 9개 열이 모두 범주 ID (숫자 크기에 의미 없음)."""

from __future__ import annotations

import pandas as pd

from km.features import auto_types, split


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train, test], keys=["train", "test"]).drop(columns=["id", "ACTION"], errors="ignore")
    df, cat_cols = auto_types(df.astype(str))
    return *split(df), cat_cols
