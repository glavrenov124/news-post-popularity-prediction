from pathlib import Path

from src.ml.data.loader import load_raw_posts
from src.ml.data.preprocessing import base_preprocessing
from src.ml.features.build_features import make_features
from src.ml.pipeline.checkpoints import log_data_ready_checkpoint


def run_prepare_data_pipeline(
    raw_data_path: str | Path,
    output_path: str | Path,
) -> Path:
    raw_data_path = Path(raw_data_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_posts(raw_data_path)
    df = base_preprocessing(df)
    df = make_features(df)

    log_data_ready_checkpoint(df)

    df.to_csv(output_path, index=False)

    print(f"Prepared dataset saved to: {output_path}")
    return output_path
