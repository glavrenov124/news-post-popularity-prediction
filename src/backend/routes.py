from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src.backend.prediction import predict_single_post
from src.backend.schemas import (
    ClipScoreResponse,
    ImproveTextRequest,
    ImproveTextResponse,
    PingResponse,
    PredictRequest,
    PredictResponse,
)
from src.backend.text_improvement import improve_text_with_llm
from src.ml.embeddings.clip import compute_clip_score, interpret_clip_score


router = APIRouter()


@router.get("/ping", response_model=PingResponse)
def ping() -> PingResponse:
    return PingResponse(status="ok")


def run_prediction(
    request: Request,
    payload: PredictRequest,
    target_name: str,
) -> PredictResponse:
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
        return PredictResponse(prediction=round(prediction, 2))

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
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
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty.")

        if not image.content_type or not image.content_type.startswith("image/"):
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

        return ClipScoreResponse(
            clip_score=round(clip_score, 2),
            label=label,
            message=message,
        )

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/improve/text", response_model=ImproveTextResponse)
def improve_text(
    request: Request,
    payload: ImproveTextRequest,
) -> ImproveTextResponse:
    try:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty.")

        if request.app.state.llm_client is None:
            raise HTTPException(
                status_code=500,
                detail="LLM client is not configured.",
            )

        improved_text, used_prompt = improve_text_with_llm(
            client=request.app.state.llm_client,
            folder_id=request.app.state.llm_folder_id,
            model_name=request.app.state.llm_model_name,
            text=payload.text,
            recommendations=payload.recommendations,
        )

        return ImproveTextResponse(
            improved_text=improved_text,
            used_prompt=used_prompt,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")