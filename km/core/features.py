"""대회 공통 피처 유틸."""

from __future__ import annotations

import pandas as pd


def auto_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """문자열/불리언 컬럼은 category, 나머지는 float 로 변환. (df, cat_cols) 반환."""
    df = df.copy()
    cat_cols = [c for c in df.columns
                if df[c].dtype == object or df[c].dtype == bool or str(df[c].dtype) in ("string", "str", "category")]
    for c in cat_cols:
        df[c] = df[c].astype(str).where(df[c].notna()).astype("category")
    for c in df.columns.difference(cat_cols):
        df[c] = df[c].astype(float)
    return df, cat_cols


def split(df: pd.DataFrame):
    """build() 안에서 concat(keys=["train","test"]) 한 df 를 다시 나눔."""
    return df.loc["train"].reset_index(drop=True), df.loc["test"].reset_index(drop=True)
