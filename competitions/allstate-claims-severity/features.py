"""Allstate 피처: 범주형 cat1~cat116 (문자열) + 연속형 cont1~cont14 그대로."""

from __future__ import annotations

import pandas as pd

from km.features import auto_types, split


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train, test], keys=["train", "test"]).drop(columns=["id", "loss"], errors="ignore")
    df, cat_cols = auto_types(df)
    return *split(df), cat_cols
