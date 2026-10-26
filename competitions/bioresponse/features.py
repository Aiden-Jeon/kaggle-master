"""Bioresponse 피처: 분자 기술자 1,776개 (모두 수치) 그대로."""

from __future__ import annotations

import pandas as pd

from km.core.features import auto_types, split


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train, test], keys=["train", "test"]).drop(columns=["Activity"], errors="ignore")
    df, cat_cols = auto_types(df)
    return *split(df), cat_cols
