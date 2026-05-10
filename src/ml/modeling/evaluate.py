def evaluate_target_predictions(y_true, y_pred, target_config) -> dict:
    postprocess = target_config.prediction_postprocess
    metrics_fn = target_config.metrics_fn

    y_true_eval = postprocess(y_true)
    y_pred_eval = postprocess(y_pred)

    return metrics_fn(y_true_eval, y_pred_eval)
