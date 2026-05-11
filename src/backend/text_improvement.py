import openai

BASE_INSTRUCTIONS = (
    "Ты редактор русскоязычных постов для соцсетей. "
    "Исправляй орфографию, пунктуацию, грамматику и стилистику. "
    "Сохраняй исходный смысл текста. "
    "Не добавляй новых фактов от себя. "
    "Если текст уже хороший, вноси только минимальные правки. "
    "Верни только итоговый улучшенный текст без комментариев."
)


def build_llm_input(text: str, recommendations: list[str]) -> str:
    filtered_recommendations = []

    for rec in recommendations:
        rec_lower = rec.lower()

        if "без изображения" in rec_lower:
            continue
        if "изображений довольно много" in rec_lower:
            continue
        if "существенных улучшений" in rec_lower:
            continue

        filtered_recommendations.append(rec)

    if not filtered_recommendations:
        return (
            "Исправь текст поста с минимальными правками. "
            "Сохрани смысл и не добавляй новых фактов.\n\n"
            f"Текст поста:\n{text}"
        )

    recommendations_text = "\n".join(f"- {rec}" for rec in filtered_recommendations)

    return (
        "Отредактируй текст поста с учётом рекомендаций ниже. "
        "Сохрани исходный смысл и не добавляй новых фактов.\n\n"
        f"Рекомендации:\n{recommendations_text}\n\n"
        f"Текст поста:\n{text}"
    )


def create_llm_client(
    api_key: str,
    folder_id: str,
) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://ai.api.cloud.yandex.net/v1",
        project=folder_id,
    )


def improve_text_with_llm(
    client: openai.OpenAI,
    folder_id: str,
    model_name: str,
    text: str,
    recommendations: list[str],
    temperature: float = 0.2,
    max_output_tokens: int = 500,
) -> tuple[str, str]:
    llm_input = build_llm_input(text, recommendations)

    response = client.responses.create(
        model=f"gpt://{folder_id}/{model_name}",
        temperature=temperature,
        instructions=BASE_INSTRUCTIONS,
        input=llm_input,
        max_output_tokens=max_output_tokens,
    )

    return response.output_text, llm_input
