import holidays
import pandas as pd


def get_is_holiday(dt_msk: str) -> int:
    dt = pd.to_datetime(dt_msk, errors="coerce")
    if pd.isna(dt):
        return 0

    ru_holidays = holidays.Russia(years=[dt.year])
    return int(dt.date() in ru_holidays)