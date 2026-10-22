"""Give Me Some Credit 피처: 원본 10개 수치 열 그대로 (결측은 MonthlyIncome, NumberOfDependents)."""

from __future__ import annotations

import pandas as pd

from km.features import auto_types, split


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train, test], keys=["train", "test"]).drop(columns=["Unnamed: 0", "SeriousDlqin2yrs"])
    df, cat_cols = auto_types(df)
    return *split(df), cat_cols
