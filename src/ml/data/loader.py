from pathlib import Path
import pandas as pd


def load_raw_posts(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
