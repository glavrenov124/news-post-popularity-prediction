import pandas as pd
from catboost import CatBoostRegressor

from src.ml.embeddings.rubert import RuBertEmbedder
from src.ml.features.build_features import make_features
from src.utils.dataframes import build_single_row_dataframe
from src.utils.dates import get_is_holiday


def build_single_text_embedding(
    embedder: RuBertEmbedder,
    text: str,
    config,
) -> pd.DataFrame:
    embedding = embedder.encode_texts(
        [text],
        batch_size=config.embeddings.batch_size,
        max_length=config.embeddings.max_length,
    )
    emb_cols = [f"emb_{i}" for i in range(embedding.shape[1])]
    return pd.DataFrame(embedding, columns=emb_cols)


def load_catboost_model(target_name: str, config) -> CatBoostRegressor:
    if target_name not in config.targets:
        raise ValueError(f"Unknown target_name: {target_name}")

    model_path = config.paths.models_dir / config.targets[target_name].model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model


def load_models(config) -> dict[str, CatBoostRegressor]:
    return {
        "views": load_catboost_model("views", config),
        "engagement_rate": load_catboost_model("engagement_rate", config),
    }


def postprocess_prediction(target_name: str, y_pred: float, config) -> float:
    target_config = config.targets[target_name]
    return float(target_config.prediction_postprocess(y_pred))


def predict_single_post(
    text: str,
    domain: str,
    dt_msk: str,
    n_photos: int,
    target_name: str,
    model: CatBoostRegressor,
    embedder: RuBertEmbedder,
    config,
) -> float:
    df = build_single_row_dataframe(
        text=text,
        domain=domain,
        dt_msk=dt_msk,
        n_photos=n_photos,
        is_holiday=get_is_holiday(dt_msk),
    )

    df = make_features(df)

    emb_df = build_single_text_embedding(
        embedder=embedder,
        text=text,
        config=config,
    )
    df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

    emb_cols = [col for col in df.columns if col.startswith("emb_")]
    feature_cols = config.features.num_cols + config.features.cat_cols + emb_cols
    df = df[feature_cols].copy()

    y_pred = model.predict(df)[0]
    return postprocess_prediction(target_name, y_pred, config)