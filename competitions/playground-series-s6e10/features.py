"""S6E10 항공 만족도 피처: 원본 + 총 지연."""

from __future__ import annotations

import pandas as pd

from km.core.features import auto_types, split


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train, test], keys=["train", "test"]).drop(columns=["id", "satisfaction"], errors="ignore")
    df["Total Delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df, cat_cols = auto_types(df)
    return *split(df), cat_cols
