from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from fin_copilot.config import DATA_RAW, load_config
from fin_copilot.data.generate import generate_panel

PARQUET = DATA_RAW / "finance_panel.parquet"


def _source() -> str:
    return os.getenv("FINCOPILOT_DATA_SOURCE", "synthetic").lower()


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
    source = _source()
    if source == "rossmann":
        from fin_copilot.data.rossmann import load as load_rossmann
        return load_rossmann()
    if source == "online_retail":
        from fin_copilot.data.online_retail import load as load_online_retail
        return load_online_retail()
    if source != "synthetic":
        raise ValueError(f"Unknown FINCOPILOT_DATA_SOURCE: {source}")
    if not path.exists():
        return build_and_save(path)
    return pd.read_parquet(path)
