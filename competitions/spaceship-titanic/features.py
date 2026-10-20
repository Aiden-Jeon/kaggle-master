"""Spaceship Titanic 피처 엔지니어링.

계약: build(train, test) -> (X_train, X_test, cat_cols)
  - 범주형 컬럼은 pandas category dtype (결측은 NaN 유지)
  - 수치형은 float, 결측은 NaN 유지 (결측 처리는 각 모델 래퍼/모델에 맡김)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATS = ["HomePlanet", "Destination", "Deck", "Side"]


def build(train: pd.DataFrame, test: pd.DataFrame):
    # 그룹/가족 크기 같은 집계 피처는 train+test 전체로 계산
    df = pd.concat([train, test], keys=["train", "test"])

    group = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = group.map(group.value_counts()).astype(float)

    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"], df["CabinNum"], df["Side"] = cabin[0], cabin[1].astype(float), cabin[2]

    surname = df["Name"].str.split().str[-1]
    df["FamilySize"] = surname.map(surname.value_counts()).astype(float)

    df["TotalSpend"] = df[SPEND].sum(axis=1, min_count=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(float)
    for c in SPEND + ["TotalSpend"]:
        df[c] = np.log1p(df[c])

    for c in ["CryoSleep", "VIP"]:
        df[c] = df[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    for c in CATS:
        df[c] = df[c].astype("category")

    cols = CATS + ["CryoSleep", "VIP", "Age", "CabinNum", "GroupSize", "FamilySize", "TotalSpend", "NoSpend"] + SPEND
    return df.loc["train", cols].reset_index(drop=True), df.loc["test", cols].reset_index(drop=True), CATS
