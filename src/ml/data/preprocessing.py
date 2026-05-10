import re

import holidays
import numpy as np
import pandas as pd


def filter_out_iqr(group: pd.DataFrame) -> pd.DataFrame:
    q1 = group["views"].quantile(0.25)
    q3 = group["views"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return group[(group["views"] >= lower) & (group["views"] <= upper)]


def clean_text_minimal(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"<.*?>", " ", text)      # убрать html
    text = re.sub(r"\r\n?", "\n", text)     # привести переносы к \n
    text = re.sub(r"[ \t]+", " ", text)     # схлопнуть пробелы и табы
    text = re.sub(r"\n{2,}", "\n", text)    # схлопнуть много пустых строк
    text = text.strip()

    return text


def base_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].apply(clean_text_minimal)
    df["dt_msk"] = pd.to_datetime(df["dt_msk"], errors="coerce")

    years = df["dt_msk"].dt.year.dropna().astype(int).unique().tolist()
    ru_holidays = holidays.Russia(years=years)
    df["is_holiday"] = df["dt_msk"].dt.date.apply(
            lambda x: int(x in ru_holidays) if pd.notna(x) else 0
        )

    if {"domain", "views"}.issubset(df.columns):
        df = (
            df.groupby("domain", group_keys=False)
            .apply(filter_out_iqr)
            .reset_index(drop=True)
        )

    df = df.query("text != ''").copy()

    df["target_views"] = np.log1p(df["views"])

    df["engagement_rate"] = (
        (df["likes"].fillna(0) + df["comments"].fillna(0) + df["reposts"].fillna(0))
        / df["views"].replace(0, np.nan)
        * 100
    )

    drop_cols = [
        "post_type",
        "is_donut",
        "post_url",
        "marked_as_ads",
        "post_source_type",
        "views",
        "likes",
        "comments",
        "reposts",
        "reactions_total",
        "owner_id",
        "post_key",
        "from_id",
        "date_unix",
        "age_days",
        "cover_photo_url",
        "all_photo_urls",
        "edited_dt_msk",
        "is_pinned"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    return df
