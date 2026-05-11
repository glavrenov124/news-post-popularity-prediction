from pathlib import Path

import pandas as pd

from src.ml.embeddings.rubert import build_rubert_embeddings


def run_prepare_embeddings_pipeline(
    prepared_data_path: str | Path,
    embeddings_output_path: str | Path,
    config,
) -> Path:
    prepared_data_path = Path(prepared_data_path)
    embeddings_output_path = Path(embeddings_output_path)
    embeddings_output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(prepared_data_path)

    emb_df = build_rubert_embeddings(
        df=df,
        text_col=config.features.text_col,
        batch_size=config.embeddings.batch_size,
        max_length=config.embeddings.max_length,
        model_name=config.embeddings.model_name,
    )

    emb_df.to_csv(embeddings_output_path, index=False)

    print(f"Embeddings saved to: {embeddings_output_path}")
    print(f"Embeddings shape: {emb_df.shape}")

    return embeddings_output_path
