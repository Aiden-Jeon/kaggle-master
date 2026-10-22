"""House Prices 피처 엔지니어링 (최소 버전).

계약: build(train, test) -> (X_train, X_test, cat_cols)
"""

from __future__ import annotations

import pandas as pd

from km.features import auto_types, split

DROP = ["Id", "SalePrice"]


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train, test], keys=["train", "test"]).drop(columns=DROP, errors="ignore")

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    # MSSubClass 는 숫자로 인코딩된 범주
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    df, cat_cols = auto_types(df)
    return *split(df), cat_cols
