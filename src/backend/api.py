from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from src.backend.logging_config import setup_logging
from src.backend.prediction import load_models
from src.backend.routes import router
from src.backend.text_improvement import create_llm_client
from src.ml.config import create_config
from src.ml.embeddings.clip import load_clip_models
from src.ml.embeddings.rubert import RuBertEmbedder
from src.utils.recommendation_rules import load_feature_thresholds_by_domain

setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config = create_config(PROJECT_ROOT)

YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER", "")
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY", "")
YANDEX_CLOUD_MODEL = os.getenv("YANDEX_CLOUD_MODEL", "yandexgpt-5.1/latest")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting backend application")

    try:
        app.state.config = config
        logger.info("Config loaded")

        logger.info("Loading RuBert embedder")
        app.state.embedder = RuBertEmbedder(
            model_name=config.embeddings.model_name,
        )
        logger.info("RuBert embedder loaded")

        logger.info("Loading prediction models")
        app.state.models = load_models(config)
        logger.info("Prediction models loaded")

        logger.info("Loading recommendation thresholds")
        app.state.thresholds_store = load_feature_thresholds_by_domain(
            str(config.paths.prepared_data_path)
        )
        logger.info("Recommendation thresholds loaded")

        logger.info("Loading CLIP models")
        clip_text_model, clip_image_model, clip_device = load_clip_models()
        app.state.clip_text_model = clip_text_model
        app.state.clip_image_model = clip_image_model
        app.state.clip_device = clip_device
        logger.info("CLIP models loaded, device=%s", clip_device)

        if YANDEX_CLOUD_API_KEY and YANDEX_CLOUD_FOLDER:
            logger.info("Configuring LLM client")
            app.state.llm_client = create_llm_client(
                api_key=YANDEX_CLOUD_API_KEY,
                folder_id=YANDEX_CLOUD_FOLDER,
            )
            app.state.llm_folder_id = YANDEX_CLOUD_FOLDER
            app.state.llm_model_name = YANDEX_CLOUD_MODEL
            logger.info("LLM client configured")
        else:
            app.state.llm_client = None
            app.state.llm_folder_id = None
            app.state.llm_model_name = None
            logger.warning("LLM client is not configured")

        logger.info("Backend startup completed")
        yield

    except Exception:
        logger.exception("Backend startup failed")
        raise

    finally:
        logger.info("Backend shutdown")


app = FastAPI(
    title="News Post Popularity API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
