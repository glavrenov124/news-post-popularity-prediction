import pandas as pd
from sklearn.model_selection import train_test_split


def make_train_valid_test_split(
    df: pd.DataFrame,
    test_size: float,
    valid_size_from_temp: float,
    random_state: int,
    shuffle: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=valid_size_from_temp,
        random_state=random_state,
        shuffle=shuffle,
    )

    return train_df.copy(), valid_df.copy(), test_df.copy()


def prepare_splits_for_target(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    text_col: str,
    num_cols: list[str],
    cat_cols: list[str],
) -> dict:
    feature_cols = [text_col] + num_cols + cat_cols

    return {
        "X_train": train_df[feature_cols].copy(),
        "y_train": train_df[target_col].values,
        "X_valid": valid_df[feature_cols].copy(),
        "y_valid": valid_df[target_col].values,
        "X_test": test_df[feature_cols].copy(),
        "y_test": test_df[target_col].values,
    }
