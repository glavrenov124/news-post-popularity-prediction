import mlflow


def log_run_to_mlflow(
    target_name: str,
    model_params: dict,
    train_metrics: dict,
    valid_metrics: dict,
    test_metrics: dict,
    model_path: str,
    dataset_shape: tuple[int, int],
    config,
):
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"catboost_{target_name}"):
        mlflow.log_param("target_name", target_name)
        mlflow.log_param("n_rows", dataset_shape[0])
        mlflow.log_param("n_cols", dataset_shape[1])

        for key, value in model_params.items():
            mlflow.log_param(key, value)

        for key, value in train_metrics.items():
            mlflow.log_metric(f"train_{key}", value)

        for key, value in valid_metrics.items():
            mlflow.log_metric(f"valid_{key}", value)

        for key, value in test_metrics.items():
            mlflow.log_metric(f"test_{key}", value)

        mlflow.log_artifact(str(model_path), artifact_path="models")
