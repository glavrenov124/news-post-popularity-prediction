from pathlib import Path

import pandas as pd

from src.ml.modeling.evaluate import evaluate_target_predictions
from src.ml.modeling.logging import log_run_to_mlflow
from src.ml.modeling.save import save_catboost_model
from src.ml.modeling.split import make_train_valid_test_split
from src.ml.modeling.train import train_catboost_regressor


def run_training_pipeline(
    prepared_data_path: str | Path,
    embeddings_path: str | Path,
    artifacts_dir: str | Path,
    target_name: str,
    config,
) -> dict:
    prepared_data_path = Path(prepared_data_path)
    embeddings_path = Path(embeddings_path)
    artifacts_dir = Path(artifacts_dir)

    if target_name not in config.targets:
        raise ValueError(f"Unknown target_name: {target_name}")

    df = pd.read_csv(prepared_data_path)
    emb_df = pd.read_csv(embeddings_path)

    df = df.merge(emb_df, on="row_id", how="left")
    emb_cols = [col for col in df.columns if col.startswith("emb_")]

    feature_cols = config.features.num_cols + config.features.cat_cols + emb_cols

    train_df, valid_df, test_df = make_train_valid_test_split(
        df=df,
        test_size=config.split.test_size,
        valid_size_from_temp=config.split.valid_size_from_temp,
        random_state=config.split.random_state,
        shuffle=config.split.shuffle,
    )

    target_config = config.targets[target_name]
    target_col = target_config.target_col
    model_params = config.training.model_params[target_name]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[target_col].values

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].values

    model = train_catboost_regressor(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        text_features=[],
        cat_features=config.features.cat_cols,
        model_params=model_params,
    )

    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)

    train_metrics = evaluate_target_predictions(y_train, train_pred, target_config)
    valid_metrics = evaluate_target_predictions(y_valid, valid_pred, target_config)
    test_metrics = evaluate_target_predictions(y_test, test_pred, target_config)

    model_path = save_catboost_model(
        model=model,
        model_dir=artifacts_dir,
        filename=target_config.model_filename,
    )

    log_run_to_mlflow(
        target_name=target_name,
        model_params={
            **model_params,
            "embedding_model": config.embeddings.model_name,
            "n_embedding_features": len(emb_cols),
        },
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        model_path=str(model_path),
        dataset_shape=df.shape,
        config=config,
    )

    return {
        "target_name": target_name,
        "pipeline_type": "embeddings",
        "model_path": model_path,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }