"""S4E2 비만도 피처: 원본 + BMI."""

from __future__ import annotations

import pandas as pd

from km.core.features import auto_types, split


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train, test], keys=["train", "test"]).drop(columns=["id", "NObeyesdad"], errors="ignore")
    df["BMI"] = df["Weight"] / df["Height"] ** 2
    df, cat_cols = auto_types(df)
    return *split(df), cat_cols
