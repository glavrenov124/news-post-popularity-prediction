import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src.backend.prediction import predict_single_post
from src.backend.recommendations import get_recommendations_for_post
from src.backend.schemas import (
    ClipScoreResponse,
    ImproveTextRequest,
    ImproveTextResponse,
    PingResponse,
    PredictRequest,
    PredictResponse,
    RecommendationsRequest,
    RecommendationsResponse,
)
from src.backend.text_improvement import improve_text_with_llm
from src.ml.embeddings.clip import compute_clip_score, interpret_clip_score

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/ping", response_model=PingResponse)
def ping() -> PingResponse:
    logger.info("Ping request received")
    return PingResponse(status="ok")


def run_prediction(
    request: Request,
    payload: PredictRequest,
    target_name: str,
) -> PredictResponse:
    started_at = time.perf_counter()
    text_length = len(payload.text.strip()) if payload.text else 0

    logger.info(
        "Prediction request started: target=%s, domain=%s, text_length=%s, n_photos=%s",
        target_name,
        payload.domain,
        text_length,
        payload.n_photos,
    )

    try:
        model = request.app.state.models[target_name]
        embedder = request.app.state.embedder
        config = request.app.state.config

        prediction = predict_single_post(
            text=payload.text,
            domain=payload.domain,
            dt_msk=payload.dt_msk,
            n_photos=payload.n_photos,
            target_name=target_name,
            model=model,
            embedder=embedder,
            config=config,
        )

        duration = time.perf_counter() - started_at
        logger.info(
            "Prediction request completed: target=%s, duration=%.3f sec",
            target_name,
            duration,
        )
        return PredictResponse(prediction=round(prediction, 2))

    except FileNotFoundError as e:
        logger.exception(
            "Prediction request failed: model file not found, target=%s",
            target_name,
        )
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.exception(
            "Prediction request failed: invalid value, target=%s",
            target_name,
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception(
            "Prediction request failed unexpectedly: target=%s",
            target_name,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/predict/views", response_model=PredictResponse)
def predict_views(request: Request, payload: PredictRequest) -> PredictResponse:
    return run_prediction(request, payload, "views")


@router.post("/predict/engagement-rate", response_model=PredictResponse)
def predict_engagement_rate(
    request: Request,
    payload: PredictRequest,
) -> PredictResponse:
    return run_prediction(request, payload, "engagement_rate")


@router.post("/analyze/clip-score", response_model=ClipScoreResponse)
async def analyze_clip_score(
    request: Request,
    text: str = Form(...),
    image: UploadFile = File(...),
) -> ClipScoreResponse:
    started_at = time.perf_counter()
    text_length = len(text.strip()) if text else 0

    logger.info(
        "CLIP request started: text_length=%s, filename=%s, content_type=%s",
        text_length,
        image.filename,
        image.content_type,
    )

    try:
        if not text.strip():
            logger.warning("CLIP request rejected: empty text")
            raise HTTPException(status_code=400, detail="Text must not be empty.")

        if not image.content_type or not image.content_type.startswith("image/"):
            logger.warning(
                "CLIP request rejected: invalid content_type=%s",
                image.content_type,
            )
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must be an image.",
            )

        image_bytes = await image.read()

        clip_score = compute_clip_score(
            text=text,
            image_bytes=image_bytes,
            text_model=request.app.state.clip_text_model,
            image_model=request.app.state.clip_image_model,
            device=request.app.state.clip_device,
        )

        label, message = interpret_clip_score(clip_score)

        duration = time.perf_counter() - started_at
        logger.info(
            "CLIP request completed: duration=%.3f sec, label=%s",
            duration,
            label,
        )

        return ClipScoreResponse(
            clip_score=round(clip_score, 2),
            label=label,
            message=message,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("CLIP request failed unexpectedly")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/recommend/text", response_model=RecommendationsResponse)
def recommend_text(
    request: Request,
    payload: RecommendationsRequest,
) -> RecommendationsResponse:
    started_at = time.perf_counter()
    text_length = len(payload.text.strip()) if payload.text else 0

    logger.info(
        "Recommendation request started: domain=%s, text_length=%s, n_photos=%s",
        payload.domain,
        text_length,
        payload.n_photos,
    )

    try:
        if not payload.text.strip():
            logger.warning("Recommendation request rejected: empty text")
            raise HTTPException(status_code=400, detail="Text must not be empty.")

        recommendations = get_recommendations_for_post(
            text=payload.text,
            domain=payload.domain,
            dt_msk=payload.dt_msk,
            n_photos=payload.n_photos,
            thresholds_store=request.app.state.thresholds_store,
        )

        duration = time.perf_counter() - started_at
        logger.info(
            "Recommendation request completed: duration=%.3f sec, recommendations_count=%s",
            duration,
            len(recommendations),
        )

        return RecommendationsResponse(recommendations=recommendations)

    except HTTPException:
        raise
    except ValueError as e:
        logger.exception("Recommendation request failed: invalid value")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Recommendation request failed unexpectedly")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/improve/text", response_model=ImproveTextResponse)
def improve_text(
    request: Request,
    payload: ImproveTextRequest,
) -> ImproveTextResponse:
    started_at = time.perf_counter()
    text_length = len(payload.text.strip()) if payload.text else 0

    logger.info(
        "Text improvement request started: domain=%s, text_length=%s, recommendations_count=%s",
        payload.domain,
        text_length,
        len(payload.recommendations),
    )

    try:
        if not payload.text.strip():
            logger.warning("Text improvement request rejected: empty text")
            raise HTTPException(status_code=400, detail="Text must not be empty.")

        if request.app.state.llm_client is None:
            logger.warning("Text improvement skipped: LLM client is not configured")
            return ImproveTextResponse(
                available=False,
                improved_text=None,
                used_prompt=None,
                message="Автоматическое улучшение текста сейчас недоступно.",
            )

        improved_text, used_prompt = improve_text_with_llm(
            client=request.app.state.llm_client,
            folder_id=request.app.state.llm_folder_id,
            model_name=request.app.state.llm_model_name,
            text=payload.text,
            recommendations=payload.recommendations,
        )

        duration = time.perf_counter() - started_at
        logger.info(
            "Text improvement request completed: duration=%.3f sec",
            duration,
        )

        return ImproveTextResponse(
            available=True,
            improved_text=improved_text,
            used_prompt=used_prompt,
            message=None,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.exception("Text improvement request failed: invalid value")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Text improvement request failed unexpectedly")
        raise HTTPException(status_code=500, detail="Internal server error")
