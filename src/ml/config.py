from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


def identity_postprocess(y):
    return y


def expm1_postprocess(y):
    return np.expm1(y)


def metrics_views(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def metrics_engagement_rate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


@dataclass(slots=True)
class PathsConfig:
    project_root: Path
    raw_data_path: Path
    prepared_data_path: Path
    embeddings_path: Path
    models_dir: Path
    mlruns_dir: Path


@dataclass(slots=True)
class FeatureConfig:
    text_col: str = "text"

    num_cols: list[str] = field(
        default_factory=lambda: [
            "n_photos",
            "hour",
            "weekday",
            "month",
            "day",
            "is_weekend",
            "is_workday",
            "is_friday",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "text_len_chars",
            "text_len_words",
            "num_lines",
            "num_exclam",
            "num_question",
            "num_dots",
            "num_commas",
            "num_colons",
            "num_hashtags",
            "num_links",
            "has_mention",
            "num_emojis",
            "caps_ratio",
            "unique_words_count",
            "unique_words_ratio",
            "avg_word_len",
            "long_words_count",
            "digit_count",
            "digit_ratio",
            "uppercase_words_count",
            "sentence_count",
            "avg_sentence_len_words",
            "ellipsis_count",
            "repeat_punct_count",
            "has_url",
            "has_number",
            "has_price",
            "starts_with_question",
            "starts_with_number",
            "starts_with_emoji",
            "has_call_to_action",
            "has_urgency",
            "is_holiday",
        ]
    )

    cat_cols: list[str] = field(
        default_factory=lambda: [
            "domain",
            "part_of_day",
            "text_len_bin",
            "hour_weekday",
            "domain_weekday",
            "domain_part_of_day",
            "weekend_hour",
        ]
    )


@dataclass(slots=True)
class SplitConfig:
    test_size: float = 0.30
    valid_size_from_temp: float = 0.50
    shuffle: bool = True
    random_state: int = 43


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "cointegrated/rubert-tiny2"
    batch_size: int = 32
    max_length: int = 512


@dataclass(slots=True)
class MlflowConfig:
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "news_post_popularity"


@dataclass(slots=True)
class TargetConfig:
    target_col: str
    prediction_postprocess: Callable[[Any], Any]
    metrics_fn: Callable[[Any, Any], dict[str, float]]
    model_filename: str


@dataclass(slots=True)
class TrainingConfig:
    model_params: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "views": {
                "iterations": 5000,
                "learning_rate": 0.03,
                "depth": 6,
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "random_seed": 42,
                "verbose": 100,
            },
            "engagement_rate": {
                "iterations": 5000,
                "learning_rate": 0.03,
                "depth": 6,
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "random_seed": 42,
                "verbose": 100,
            },
        }
    )


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    targets: dict[str, TargetConfig] = field(
        default_factory=lambda: {
            "views": TargetConfig(
                target_col="target_views",
                prediction_postprocess=expm1_postprocess,
                metrics_fn=metrics_views,
                model_filename="catboost_views.cbm",
            ),
            "engagement_rate": TargetConfig(
                target_col="engagement_rate",
                prediction_postprocess=identity_postprocess,
                metrics_fn=metrics_engagement_rate,
                model_filename="catboost_engagement_rate.cbm",
            ),
        }
    )


def create_config(project_root: str | Path) -> AppConfig:
    project_root = Path(project_root)

    paths = PathsConfig(
        project_root=project_root,
        raw_data_path=project_root / "data" / "raw" / "posts_vk.csv",
        prepared_data_path=project_root / "artifacts" / "data" / "prepared_features.csv",
        embeddings_path=project_root / "artifacts" / "data" / "embeddings.csv",
        models_dir=project_root / "artifacts" / "models",
        mlruns_dir=project_root / "mlruns",
    )

    return AppConfig(paths=paths)