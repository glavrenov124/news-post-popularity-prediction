from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
import hashlib
import os 
import pandas as pd
import requests
import streamlit as st

from recommendations import (
    generate_recommendations,
    get_thresholds_for_domain,
    load_feature_thresholds_by_domain,
)
from src.ml.features.build_features import make_features


st.set_page_config(page_title="News Post Popularity", layout="centered")


API_BASE_URL = os.getenv("API_BASE_URL") or "http://127.0.0.1:8000"
MIN_TEXT_LENGTH = 20
PREPARED_FEATURES_PATH = PROJECT_ROOT / "artifacts" / "data" / "prepared_features.csv"

DOMAINS = [
    "ndnews24",
    "literabook",
    "nrmusicru",
    "lentach",
    "unknown",
]


@st.cache_resource
def get_thresholds_store() -> dict:
    return load_feature_thresholds_by_domain(str(PREPARED_FEATURES_PATH))


def build_single_row_dataframe(
    text: str,
    domain: str,
    dt_msk: str,
    n_photos: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": text,
                "domain": domain,
                "dt_msk": dt_msk,
                "n_photos": n_photos,
                "is_pinned": 0,
                "row_id": 0,
            }
        ]
    )


def build_dt_msk(use_now: bool, selected_date, selected_time) -> str:
    if use_now:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    combined = datetime.combine(selected_date, selected_time)
    return combined.strftime("%Y-%m-%d %H:%M:%S")


def call_prediction(endpoint: str, payload: dict) -> float:
    response = requests.post(
        f"{API_BASE_URL}{endpoint}",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["prediction"]


def call_clip_score(text: str, image_file) -> dict:
    files = {
        "image": (
            image_file.name,
            image_file.getvalue(),
            image_file.type,
        )
    }
    data = {
        "text": text,
    }

    response = requests.post(
        f"{API_BASE_URL}/analyze/clip-score",
        data=data,
        files=files,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def call_text_improvement(
    text: str,
    domain: str,
    recommendations: list[str],
) -> dict:
    payload = {
        "text": text,
        "domain": domain,
        "recommendations": recommendations,
    }

    response = requests.post(
        f"{API_BASE_URL}/improve/text",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def make_file_id(uploaded_file) -> str:
    content = uploaded_file.getvalue()
    raw = f"{uploaded_file.name}_{len(content)}".encode("utf-8") + content[:100]
    return hashlib.md5(raw).hexdigest()


def init_session_state() -> None:
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = {}

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def sync_uploaded_files(uploaded_files) -> None:
    if not uploaded_files:
        return

    added_new_file = False

    for uploaded_file in uploaded_files:
        file_id = make_file_id(uploaded_file)

        if file_id not in st.session_state.uploaded_images:
            st.session_state.uploaded_images[file_id] = {
                "id": file_id,
                "name": uploaded_file.name,
                "file": uploaded_file,
            }
            added_new_file = True

    if added_new_file:
        st.session_state.uploader_key += 1
        st.rerun()


def remove_image(file_id: str) -> None:
    if file_id in st.session_state.uploaded_images:
        del st.session_state.uploaded_images[file_id]


def clear_all_images() -> None:
    st.session_state.uploaded_images = {}


def render_image_gallery() -> int:
    images = list(st.session_state.uploaded_images.values())

    if not images:
        st.info("Фотографии не прикреплены.")
        return 0

    st.write(f"Прикреплено фото: {len(images)}")

    for start_idx in range(0, len(images), 3):
        row = images[start_idx:start_idx + 3]
        cols = st.columns(3)

        for col, image_data in zip(cols, row):
            with col:
                st.image(
                    image_data["file"],
                    caption=image_data["name"],
                    use_container_width=True,
                )
                if st.button("Удалить", key=f"delete_{image_data['id']}"):
                    remove_image(image_data["id"])
                    st.rerun()

    if st.button("Удалить все фото"):
        clear_all_images()
        st.rerun()

    return len(images)


def render_clip_message(clip_result: dict) -> None:
    st.subheader("Соответствие текста и изображения")

    if clip_result["label"] == "low":
        st.warning(clip_result["message"])
    elif clip_result["label"] == "medium":
        st.info(clip_result["message"])
    elif clip_result["label"] == "good":
        st.success(clip_result["message"])
    else:
        st.success(clip_result["message"])


def get_recommendations(
    text: str,
    domain: str,
    dt_msk: str,
    n_photos: int,
) -> list[str]:
    thresholds_store = get_thresholds_store()

    single_df = build_single_row_dataframe(
        text=text,
        domain=domain,
        dt_msk=dt_msk,
        n_photos=n_photos,
    )
    single_df = make_features(single_df)

    features_row = single_df.iloc[0].to_dict()
    domain_thresholds = get_thresholds_for_domain(thresholds_store, domain)

    recommendations = generate_recommendations(
        features_row=features_row,
        thresholds=domain_thresholds,
    )

    return recommendations


def render_recommendations(recommendations: list[str]) -> None:
    st.subheader("Что можно улучшить")
    for rec in recommendations:
        st.write(f"- {rec}")


def main():
    init_session_state()

    st.title("Прогноз популярности поста")
    st.write("Заполните параметры публикации и получите прогноз популярности.")

    text = st.text_area(
        "Текст поста",
        height=250,
        placeholder="Вставьте текст поста сюда...",
        help=f"Минимальная длина текста — {MIN_TEXT_LENGTH} символов.",
    )

    text_length = len(text.strip()) if text else 0
    st.caption(f"Длина текста: {text_length} символов")

    if text and text_length < MIN_TEXT_LENGTH:
        st.warning(f"Введите ещё минимум {MIN_TEXT_LENGTH - text_length} символов.")

    domain = st.selectbox(
        "Домен",
        options=DOMAINS,
        index=0,
    )

    st.subheader("Изображения")
    st.caption(
        "Можно прикрепить одну или несколько картинок. "
        "Для оценки соответствия текста и изображения используется только первая фотография."
    )

    uploaded_files = st.file_uploader(
        "Добавить фото",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    sync_uploaded_files(uploaded_files)
    n_photos = render_image_gallery()

    use_now = st.checkbox("Использовать текущее время", value=True)

    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.date_input(
            "Дата публикации",
            value=datetime.now().date(),
            disabled=use_now,
        )

    with col2:
        selected_time = st.time_input(
            "Время публикации",
            value=datetime.now().time().replace(second=0, microsecond=0),
            disabled=use_now,
        )

    if st.button("Предсказать", type="primary", use_container_width=True):
        if not text.strip():
            st.error("Нужно ввести текст поста.")
            return

        if text_length < MIN_TEXT_LENGTH:
            st.error(f"Текст должен содержать не менее {MIN_TEXT_LENGTH} символов.")
            return

        dt_msk = build_dt_msk(use_now, selected_date, selected_time)

        payload = {
            "text": text,
            "domain": domain,
            "dt_msk": dt_msk,
            "n_photos": n_photos,
        }

        try:
            with st.spinner("Считаю прогноз..."):
                views_pred = call_prediction("/predict/views", payload)
                er_pred = call_prediction("/predict/engagement-rate", payload)

                clip_result = None
                images = list(st.session_state.uploaded_images.values())

                if images:
                    first_image = images[0]["file"]
                    clip_result = call_clip_score(text, first_image)

                recommendations = get_recommendations(
                    text=text,
                    domain=domain,
                    dt_msk=dt_msk,
                    n_photos=n_photos,
                )

                improved_result = call_text_improvement(
                    text=text,
                    domain=domain,
                    recommendations=recommendations,
                )

            st.subheader("Результат прогноза")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Прогноз просмотров",
                    f"{views_pred:,.0f}".replace(",", " "),
                )

            with result_col2:
                st.metric(
                    "Прогноз engagement rate",
                    f"{er_pred:.2f}%",
                )

            if clip_result is not None:
                render_clip_message(clip_result)

            render_recommendations(recommendations)

            st.subheader("Улучшенный текст")
            st.text_area(
                "Версия после редактирования",
                value=improved_result["improved_text"],
                height=250,
            )

        except requests.exceptions.ConnectionError:
            st.error("Сервис временно недоступен. Попробуйте позже.")
        except requests.exceptions.Timeout:
            st.error("Сервис отвечает слишком долго. Попробуйте ещё раз.")
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = e.response.json().get("detail", "Ошибка при обработке запроса.")
            except Exception:
                error_detail = "Ошибка при обработке запроса."
            st.error(error_detail)
        except Exception as e:
            st.error(f"Произошла непредвиденная ошибка: {e}")


if __name__ == "__main__":
    main()