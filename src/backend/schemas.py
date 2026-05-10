from pydantic import BaseModel, Field


class PingResponse(BaseModel):
    status: str


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст поста")
    domain: str = Field(default="unknown", description="Сообщество / домен")
    dt_msk: str = Field(
        ...,
        description="Дата и время публикации в формате YYYY-MM-DD HH:MM:SS",
    )
    n_photos: int = Field(default=1, ge=0)


class PredictResponse(BaseModel):
    prediction: float


class ClipScoreResponse(BaseModel):
    clip_score: float
    label: str
    message: str


class ImproveTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст поста")
    domain: str = Field(default="unknown", description="Сообщество / домен")
    recommendations: list[str] = Field(
        default_factory=list,
        description="Список рекомендаций по улучшению текста",
    )


class ImproveTextResponse(BaseModel):
    improved_text: str
    used_prompt: str