from pathlib import Path

import click

from src.ml.config import create_config
from src.ml.pipeline.prepare_data_pipeline import run_prepare_data_pipeline
from src.ml.pipeline.prepare_embeddings_pipeline import run_prepare_embeddings_pipeline
from src.ml.pipeline.training_pipeline import run_training_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config = create_config(PROJECT_ROOT)


@click.group()
def cli():
    pass


@cli.command("prepare-data")
def prepare_data_command():
    run_prepare_data_pipeline(
        raw_data_path=config.paths.raw_data_path,
        output_path=config.paths.prepared_data_path,
    )


@cli.command("prepare-embeddings")
def prepare_embeddings_command():
    run_prepare_embeddings_pipeline(
        prepared_data_path=config.paths.prepared_data_path,
        embeddings_output_path=config.paths.embeddings_path,
        config=config,
    )


@cli.command("train")
@click.option(
    "--target",
    "target_name",
    type=click.Choice(["views", "engagement_rate"]),
    required=True,
)
def train_command(target_name: str):
    result = run_training_pipeline(
        prepared_data_path=config.paths.prepared_data_path,
        embeddings_path=config.paths.embeddings_path,
        artifacts_dir=config.paths.models_dir,
        target_name=target_name,
        config=config,
    )

    print(f"\nTarget: {result['target_name']}")
    print(f"Pipeline: {result['pipeline_type']}")
    print("Model path:", result["model_path"])
    print("Train metrics:", result["train_metrics"])
    print("Valid metrics:", result["valid_metrics"])
    print("Test metrics:", result["test_metrics"])


if __name__ == "__main__":
    cli()