from __future__ import annotations

from pathlib import Path

import pandas as pd

from fin_copilot.config import DATA_RAW, load_config
from fin_copilot.data.generate import generate_panel

PARQUET = DATA_RAW / "finance_panel.parquet"


def build_and_save(path: Path = PARQUET) -> pd.DataFrame:
    cfg = load_config()
    df = generate_panel(
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        categories=cfg.data.categories,
        regions=cfg.data.regions,
        seed=cfg.data.seed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load(path: Path = PARQUET) -> pd.DataFrame:
    if not path.exists():
        return build_and_save(path)
    return pd.read_parquet(path)
