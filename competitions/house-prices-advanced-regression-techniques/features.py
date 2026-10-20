"""House Prices 피처 엔지니어링 (최소 버전).

계약: build(train, test) -> (X_train, X_test, cat_cols)
"""

from __future__ import annotations

import pandas as pd

DROP = ["Id", "SalePrice"]


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train, test], keys=["train", "test"]).drop(columns=DROP, errors="ignore")

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    # MSSubClass 는 숫자로 인코딩된 범주
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    cat_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) in ("string", "str")]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    for c in df.columns.difference(cat_cols):
        df[c] = df[c].astype(float)

    return df.loc["train"].reset_index(drop=True), df.loc["test"].reset_index(drop=True), cat_cols
