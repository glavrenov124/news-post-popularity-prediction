from pathlib import Path


def save_catboost_model(model, model_dir: str | Path, filename: str) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / filename
    model.save_model(str(model_path))
    return model_path
