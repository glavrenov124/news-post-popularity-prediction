import pandas as pd


def build_single_row_dataframe(
    text: str,
    domain: str,
    dt_msk: str,
    n_photos: int,
    is_holiday: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": text,
                "domain": domain,
                "dt_msk": dt_msk,
                "n_photos": n_photos,
                "is_pinned": 0,
                "is_holiday": is_holiday,
                "row_id": 0,
            }
        ]
    )