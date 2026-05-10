from catboost import CatBoostRegressor


def train_catboost_regressor(
    X_train,
    y_train,
    text_features,
    cat_features,
    model_params: dict,
    X_valid=None,
    y_valid=None,
):
    model = CatBoostRegressor(**model_params)

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "text_features": text_features,
        "cat_features": cat_features,
    }

    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = (X_valid, y_valid)
        fit_kwargs["use_best_model"] = True
        fit_kwargs["early_stopping_rounds"] = 200

    model.fit(**fit_kwargs)
    return model
