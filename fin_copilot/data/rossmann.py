from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from fin_copilot.config import DATA_RAW

DATASET = "pratyushakar/rossmann-store-sales"
RAW_DIR = DATA_RAW / "rossmann"
PANEL_PATH = DATA_RAW / "rossmann_panel.parquet"

STORETYPE_NAMES = {"a": "Standard", "b": "Boutique", "c": "Convenience", "d": "Hypermarket"}
ASSORTMENT_NAMES = {"a": "Basic", "b": "Extra", "c": "Extended"}
COGS_RATE_BY_TYPE = {"Standard": 0.58, "Boutique": 0.62, "Convenience": 0.54, "Hypermarket": 0.50}


def download_raw(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if (RAW_DIR / "train.csv").exists() and (RAW_DIR / "store.csv").exists() and not force:
        return RAW_DIR
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError(
            "The 'kaggle' package is required. Install with: pip install kaggle"
        ) from e

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(RAW_DIR), quiet=False, unzip=False)

    for zip_path in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()
    return RAW_DIR


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(
        RAW_DIR / "train.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": str},
        low_memory=False,
    )
    store = pd.read_csv(RAW_DIR / "store.csv")
    return train, store


MIN_STORES_PER_GROUP = 30
MIN_COVERAGE_FRACTION = 0.5


def to_panel(save: bool = True, force: bool = False) -> pd.DataFrame:
    download_raw(force=force)
    train, store = _load_raw()

    train = train[(train["Open"] == 1) & (train["Sales"] > 0)].copy()
    merged = train.merge(store[["Store", "StoreType", "Assortment"]], on="Store", how="left")
    merged["category"] = merged["StoreType"].map(STORETYPE_NAMES)
    merged["region"] = merged["Assortment"].map(ASSORTMENT_NAMES)

    store_counts = merged.groupby(["category", "region"])["Store"].nunique()
    keep = store_counts[store_counts >= MIN_STORES_PER_GROUP].index
    merged = merged[merged.set_index(["category", "region"]).index.isin(keep)].copy()

    panel = (
        merged.groupby(["Date", "category", "region"], observed=True)
        .agg(
            revenue=("Sales", "sum"),
            units_sold=("Customers", "sum"),
            n_open=("Store", "nunique"),
        )
        .reset_index()
        .rename(columns={"Date": "date"})
    )

    typical = panel.groupby(["category", "region"])["n_open"].transform("quantile", 0.9)
    panel = panel[panel["n_open"] >= MIN_COVERAGE_FRACTION * typical].copy()

    panel["avg_price"] = (panel["revenue"] / panel["units_sold"]).round(2)
    panel["cogs"] = (panel["revenue"] * panel["category"].map(COGS_RATE_BY_TYPE)).round(2)
    panel["gross_profit"] = (panel["revenue"] - panel["cogs"]).round(2)
    panel = panel.drop(columns=["n_open"])
    panel = panel.sort_values(["category", "region", "date"]).reset_index(drop=True)

    if save:
        PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(PANEL_PATH, index=False)
    return panel


def load(force_rebuild: bool = False) -> pd.DataFrame:
    if PANEL_PATH.exists() and not force_rebuild:
        return pd.read_parquet(PANEL_PATH)
    return to_panel(save=True, force=force_rebuild)
