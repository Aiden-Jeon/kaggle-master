"""Titanic 피처: 호칭, 가족 크기, 객실 데크, 티켓 공유 수."""

from __future__ import annotations

import pandas as pd

from km.core.features import auto_types, split

TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train, test], keys=["train", "test"])

    title = df["Name"].str.extract(r",\s*([^\.]+)\.")[0].replace(TITLE_MAP)
    df["Title"] = title.where(title.map(title.value_counts()) >= 10, "Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Deck"] = df["Cabin"].str[0]
    df["HasCabin"] = df["Cabin"].notna().astype(float)
    df["TicketFreq"] = df["Ticket"].map(df["Ticket"].value_counts())

    cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "FamilySize", "Deck", "HasCabin",
            "TicketFreq"]
    df, cat_cols = auto_types(df[cols])
    return *split(df), cat_cols
