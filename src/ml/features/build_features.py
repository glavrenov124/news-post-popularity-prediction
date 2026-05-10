import re
import numpy as np
import pandas as pd


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dt = pd.to_datetime(df["dt_msk"], errors="coerce")

    df["hour"] = dt.dt.hour.fillna(0).astype("int16")
    df["weekday"] = dt.dt.weekday.fillna(0).astype("int16")
    df["month"] = dt.dt.month.fillna(0).astype("int16")
    df["day"] = dt.dt.day.fillna(0).astype("int16")

    df["is_weekend"] = (df["weekday"] >= 5).astype("int8")
    df["is_workday"] = (df["weekday"] < 5).astype("int8")
    df["is_friday"] = (df["weekday"] == 4).astype("int8")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    if "n_photos" in df.columns:
        df["n_photos"] = df["n_photos"].fillna(0).astype("int16")
    else:
        df["n_photos"] = 0

    if "is_pinned" in df.columns:
        df["is_pinned"] = df["is_pinned"].fillna(0).astype("int8")
    else:
        df["is_pinned"] = 0

    txt = df["text"].fillna("").astype(str)

    df["text_len_chars"] = txt.str.len().astype("int32")
    df["text_len_words"] = txt.str.split().str.len().fillna(0).astype("int32")
    df["num_lines"] = (txt.str.count("\n") + 1).astype("int16")

    df["num_exclam"] = txt.str.count("!").astype("int16")
    df["num_question"] = txt.str.count(r"\?").astype("int16")
    df["num_dots"] = txt.str.count(r"\.").astype("int16")
    df["num_commas"] = txt.str.count(",").astype("int16")
    df["num_colons"] = txt.str.count(":").astype("int16")

    df["num_hashtags"] = txt.str.count("#").astype("int16")
    df["num_links"] = (txt.str.count("http") + txt.str.count("vk.cc")).astype("int16")
    df["has_mention"] = txt.str.contains(r"\[club|\[id|@", regex=True, na=False).astype(
        "int8"
    )
    df["num_emojis"] = txt.str.count(r"[\U0001F300-\U0001FAFF]").astype("int16")

    letters = txt.str.findall(r"[A-Za-zА-Яа-яЁё]")
    caps = txt.str.findall(r"[A-ZА-ЯЁ]")
    df["caps_ratio"] = (
        (caps.str.len() / letters.str.len().replace(0, np.nan))
        .fillna(0.0)
        .astype("float32")
    )

    df["is_empty_text"] = (df["text_len_chars"] == 0).astype("int8")

    words = txt.str.findall(r"\w+", flags=re.UNICODE)
    digits = txt.str.findall(r"\d")
    caps_words = txt.str.findall(r"\b[A-ZА-ЯЁ]{2,}\b")

    df["unique_words_count"] = words.apply(
        lambda x: len(set(w.lower() for w in x))
    ).astype("int32")

    df["unique_words_ratio"] = (
        (df["unique_words_count"] / df["text_len_words"].replace(0, np.nan))
        .fillna(0.0)
        .astype("float32")
    )

    df["avg_word_len"] = words.apply(
        lambda x: np.mean([len(w) for w in x]) if len(x) else 0
    ).astype("float32")

    df["long_words_count"] = words.apply(lambda x: sum(len(w) >= 8 for w in x)).astype(
        "int16"
    )

    df["digit_count"] = digits.str.len().astype("int16")
    df["digit_ratio"] = (
        (df["digit_count"] / df["text_len_chars"].replace(0, np.nan))
        .fillna(0.0)
        .astype("float32")
    )

    df["uppercase_words_count"] = caps_words.str.len().astype("int16")

    sentence_count = txt.str.count(r"[.!?]+")
    df["sentence_count"] = np.where(
        df["text_len_chars"] > 0,
        sentence_count + 1,
        0,
    ).astype("int16")

    df["avg_sentence_len_words"] = (
        (df["text_len_words"] / df["sentence_count"].replace(0, np.nan))
        .fillna(0.0)
        .astype("float32")
    )

    df["ellipsis_count"] = txt.str.count(r"\.\.\.|…").astype("int16")
    df["repeat_punct_count"] = txt.str.count(r"[!?]{2,}").astype("int16")

    df["has_url"] = txt.str.contains(r"http|www|vk\.cc", regex=True, na=False).astype(
        "int8"
    )
    df["has_number"] = txt.str.contains(r"\d", regex=True, na=False).astype("int8")
    df["has_price"] = txt.str.contains(
        r"\d+\s?(₽|руб|р\b)|скидк|цена|бесплатно|акция",
        regex=True,
        case=False,
        na=False,
    ).astype("int8")

    df["starts_with_question"] = txt.str.match(
        r"^\s*[^a-zA-ZА-Яа-яЁё0-9]*.*\?", na=False
    ).astype("int8")
    df["starts_with_number"] = txt.str.match(r"^\s*\d", na=False).astype("int8")
    df["starts_with_emoji"] = txt.str.match(
        r"^\s*[\U0001F300-\U0001FAFF]", na=False
    ).astype("int8")

    cta_pattern = (
        r"пиши|смотри|читай|успей|переходи|жми|сохрани|подпишись|голосуй|оцени"
    )
    urgency_pattern = r"сегодня|сейчас|срочно|только сегодня|последний шанс|успей"

    df["has_call_to_action"] = txt.str.contains(
        cta_pattern,
        regex=True,
        case=False,
        na=False,
    ).astype("int8")

    df["has_urgency"] = txt.str.contains(
        urgency_pattern,
        regex=True,
        case=False,
        na=False,
    ).astype("int8")

    df["part_of_day"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    )

    df["text_len_bin"] = pd.cut(
        df["text_len_words"],
        bins=[-1, 5, 15, 30, 60, 10**9],
        labels=["very_short", "short", "medium", "long", "very_long"],
    )

    df["hour_weekday"] = df["hour"].astype(str) + "_" + df["weekday"].astype(str)
    df["domain_weekday"] = (
        df["domain"].fillna("unknown").astype(str) + "_" + df["weekday"].astype(str)
    )
    df["domain_part_of_day"] = (
        df["domain"].fillna("unknown").astype(str) + "_" + df["part_of_day"].astype(str)
    )
    df["weekend_hour"] = df["is_weekend"].astype(str) + "_" + df["hour"].astype(str)

    return df
