"""Restaurant Revenue 피처: 개점일 → 영업 기간/연·월, 익명 P1~P37 그대로.

City 는 학습 137행에 34개라 테스트 도시 대부분이 처음 보는 값 → 버리고 City Group 만 사용.
"""

from __future__ import annotations

import pandas as pd

from km.core.features import auto_types, split


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train, test], keys=["train", "test"]).drop(columns=["Id", "revenue", "City"], errors="ignore")
    opened = pd.to_datetime(df.pop("Open Date"), format="%m/%d/%Y")
    df["DaysOpen"] = (pd.Timestamp("2015-01-01") - opened).dt.days  # 대회 시작 시점 기준
    df["OpenYear"] = opened.dt.year
    df["OpenMonth"] = opened.dt.month
    df, cat_cols = auto_types(df)
    return *split(df), cat_cols
