import pandas as pd


def _compute_thresholds(df: pd.DataFrame) -> dict:
    return {
        "text_len_words_p25": df["text_len_words"].quantile(0.25),
        "text_len_words_p75": df["text_len_words"].quantile(0.75),
        "avg_sentence_len_words_p75": df["avg_sentence_len_words"].quantile(0.75),
        "num_hashtags_p90": df["num_hashtags"].quantile(0.90),
        "num_links_p90": df["num_links"].quantile(0.90),
        "caps_ratio_p90": df["caps_ratio"].quantile(0.90),
        "uppercase_words_count_p90": df["uppercase_words_count"].quantile(0.90),
    }


def load_feature_thresholds_by_domain(prepared_path: str) -> dict:
    df = pd.read_csv(prepared_path)
    df["domain"] = df["domain"].fillna("unknown").astype(str)

    global_thresholds = _compute_thresholds(df)

    domain_thresholds = {}
    domain_sizes = df["domain"].value_counts().to_dict()

    for domain, domain_df in df.groupby("domain"):
        domain_thresholds[domain] = _compute_thresholds(domain_df)

    return {
        "global": global_thresholds,
        "by_domain": domain_thresholds,
        "domain_sizes": domain_sizes,
    }


def get_thresholds_for_domain(
    thresholds_store: dict,
    domain: str,
) -> dict:
    domain = (domain or "unknown").strip()

    if domain in thresholds_store["by_domain"]:
        return thresholds_store["by_domain"][domain]

    return thresholds_store["global"]


def generate_recommendations(
    features_row: dict,
    thresholds: dict,
) -> list[str]:
    recommendations = []

    text_len_words = features_row.get("text_len_words", 0)
    sentence_count = features_row.get("sentence_count", 0)
    avg_sentence_len_words = features_row.get("avg_sentence_len_words", 0.0)
    num_lines = features_row.get("num_lines", 1)

    num_exclam = features_row.get("num_exclam", 0)
    num_hashtags = features_row.get("num_hashtags", 0)
    num_links = features_row.get("num_links", 0)

    caps_ratio = features_row.get("caps_ratio", 0.0)
    uppercase_words_count = features_row.get("uppercase_words_count", 0)

    has_call_to_action = features_row.get("has_call_to_action", 0)
    has_urgency = features_row.get("has_urgency", 0)
    n_photos = features_row.get("n_photos", 0)
    text_len_chars = features_row.get("text_len_chars", 0)

    if text_len_words < thresholds["text_len_words_p25"]:
        recommendations.append(
            "Текст короче типичного уровня для этого источника. Можно добавить больше деталей и контекста."
        )

    if text_len_words > thresholds["text_len_words_p75"]:
        recommendations.append(
            "Текст длиннее обычного для этого источника. Попробуйте быстрее перейти к главной мысли."
        )

    if sentence_count <= 1 and text_len_words > 20:
        recommendations.append(
            "Текст выглядит слишком цельным блоком. Разбейте его на несколько предложений или абзацев."
        )

    if avg_sentence_len_words > thresholds["avg_sentence_len_words_p75"]:
        recommendations.append(
            "Предложения длиннее типичного уровня. Более короткие формулировки улучшат читаемость."
        )

    if num_lines <= 1 and text_len_chars > 250:
        recommendations.append(
            "В тексте мало визуального деления. Разбиение на абзацы поможет восприятию."
        )

    if num_exclam >= 4:
        recommendations.append(
            "В тексте слишком много восклицательных знаков. Более нейтральная подача может выглядеть убедительнее."
        )

    if num_hashtags > thresholds["num_hashtags_p90"]:
        recommendations.append(
            "Хештегов больше, чем обычно. Лучше оставить только самые важные."
        )

    if num_links > thresholds["num_links_p90"]:
        recommendations.append(
            "Ссылок в тексте довольно много. Проверьте, что они не отвлекают от основного сообщения."
        )

    if (
        caps_ratio > thresholds["caps_ratio_p90"]
        or uppercase_words_count > thresholds["uppercase_words_count_p90"]
    ):
        recommendations.append(
            "В тексте слишком много акцентов верхним регистром. Это может ухудшать восприятие."
        )

    if has_call_to_action == 0:
        recommendations.append(
            "Подачу можно сделать чуть более вовлекающей, чтобы текст выглядел живее для читателя."
        )

    if has_urgency == 1:
        recommendations.append(
            "Если используется срочная подача, убедитесь, что в начале текста есть конкретные факты и ясный инфоповод."
        )

    if n_photos == 0:
        recommendations.append(
            "Публикация без изображения может быть менее заметной в ленте."
        )
    elif n_photos > 4:
        recommendations.append(
            "Изображений довольно много. Убедитесь, что они не размывают основной акцент публикации."
        )

    if not recommendations:
        recommendations.append(
            "Пост выглядит достаточно сильным. Существенных улучшений перед публикацией не требуется."
        )

    return recommendations
