from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.ml.features.build_features import make_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"

MODEL_FILENAMES = {
    "views": "catboost_views.cbm",
    "engagement_rate": "catboost_engagement_rate.cbm",
}


def get_is_holiday(dt_msk: str) -> int:
    dt = pd.to_datetime(dt_msk, errors="coerce")
    if pd.isna(dt):
        return 0

    ru_holidays = holidays.Russia(years=[dt.year])
    return int(dt.date() in ru_holidays)


def build_single_row_dataframe(
    text: str,
    domain: str,
    dt_msk: str,
    n_photos: int,
) -> pd.DataFrame:
    is_holiday = get_is_holiday(dt_msk)

    df = pd.DataFrame(
        [
            {
                "text": text,
                "domain": domain,
                "dt_msk": dt_msk,
                "n_photos": n_photos,
                "is_pinned": 0,
                "is_holiday": is_holiday,
            }
        ]
    )
    return df


def load_catboost_model(target_name: str) -> CatBoostRegressor:
    if target_name not in MODEL_FILENAMES:
        raise ValueError(f"Unknown target_name: {target_name}")

    model_path = MODELS_DIR / MODEL_FILENAMES[target_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model


def postprocess_prediction(target_name: str, y_pred: float) -> float:
    if target_name == "views":
        return float(np.expm1(y_pred))
    return float(y_pred)


def predict_single_post(
    text: str,
    domain: str,
    dt_msk: str,
    n_photos: int,
    target_name: str,
) -> float:
    df = build_single_row_dataframe(
        text=text,
        domain=domain,
        dt_msk=dt_msk,
        n_photos=n_photos,
    )

    df = make_features(df)

    model = load_catboost_model(target_name)
    y_pred = model.predict(df)[0]

    return postprocess_prediction(target_name, y_pred)
