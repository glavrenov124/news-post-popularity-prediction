from src.ml.features.build_features import make_features
from src.utils.dataframes import build_single_row_dataframe
from src.utils.dates import get_is_holiday
from src.utils.recommendation_rules import (
    generate_recommendations,
    get_thresholds_for_domain,
)


def get_recommendations_for_post(
    text: str,
    domain: str,
    dt_msk: str,
    n_photos: int,
    thresholds_store: dict,
) -> list[str]:
    single_df = build_single_row_dataframe(
        text=text,
        domain=domain,
        dt_msk=dt_msk,
        n_photos=n_photos,
        is_holiday=get_is_holiday(dt_msk),
    )
    single_df = make_features(single_df)

    features_row = single_df.iloc[0].to_dict()
    domain_thresholds = get_thresholds_for_domain(thresholds_store, domain)

    return generate_recommendations(
        features_row=features_row,
        thresholds=domain_thresholds,
    )