import pandas as pd


def log_data_ready_checkpoint(df: pd.DataFrame) -> None:
    print("\n Reading for training")
    print(f"Shape: {df.shape}")

    important_cols = [
        "text",
        "dt_msk",
        "domain",
        "target_views",
        "engagement_rate",
        "hour",
        "weekday",
        "text_len_chars",
        "text_len_words",
    ]

    existing_cols = [col for col in important_cols if col in df.columns]
    print("Important columns present:", existing_cols)

    if "target_views" in df.columns:
        print("target_views NaN:", int(df["target_views"].isna().sum()))

    if "engagement_rate" in df.columns:
        print("engagement_rate NaN:", int(df["engagement_rate"].isna().sum()))

    print("Пример данных:")
    print(df.head(3))
    print("Датасет успешно предобработан\n")
