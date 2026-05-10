from io import BytesIO
import re

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util


TEXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
IMAGE_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip_models() -> tuple[SentenceTransformer, SentenceTransformer, str]:
    device = get_device()

    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=device)
    image_model = SentenceTransformer(IMAGE_MODEL_NAME, device=device)

    return text_model, image_model, device


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def preprocess_text_for_clip(
    text: str | None,
    max_words: int = 30,
    max_chars: int = 200,
) -> str:
    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"\[[^\]|]+\|[^\]]+\]", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[\"'«»„“]", " ", text)
    text = re.sub(r"[-–—-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    text = " ".join(words[:max_words])
    text = text[:max_chars].strip()

    return text


@torch.no_grad()
def compute_clip_score(
    text: str,
    image_bytes: bytes,
    text_model: SentenceTransformer,
    image_model: SentenceTransformer,
    device: str,
) -> float:
    image = load_image_from_bytes(image_bytes)
    text = preprocess_text_for_clip(text)

    text_emb = text_model.encode(
        [text],
        convert_to_tensor=True,
        device=device,
        normalize_embeddings=True,
    )

    image_emb = image_model.encode(
        [image],
        convert_to_tensor=True,
        device=device,
        normalize_embeddings=True,
    )

    clip_score = util.cos_sim(text_emb, image_emb)[0][0].item() * 100
    return float(clip_score)


def interpret_clip_score(score: float) -> tuple[str, str]:
    if score < 10:
        return (
            "low",
            "Изображение слабо соответствует тексту. Лучше подобрать другую картинку.",
        )
    if score < 20:
        return (
            "medium",
            "Соответствие текста и изображения среднее.",
        )
    if score < 30:
        return (
            "good",
            "Изображение достаточно хорошо соответствует тексту.",
        )
    return (
        "excellent",
        "Изображение очень хорошо соответствует тексту.",
    )