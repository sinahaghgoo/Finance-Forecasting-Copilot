"""Loader for the UCI 'Online Retail II' dataset.

A public, directly downloadable retail transaction dataset covering a UK-based
online retailer from 2009-12 to 2011-12. Aggregated here to a daily country ×
product-category panel compatible with the rest of the pipeline.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from fin_copilot.config import DATA_RAW

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = DATA_RAW / "online_retail"
RAW_XLSX = RAW_DIR / "online_retail_II.xlsx"
PANEL_PATH = DATA_RAW / "online_retail_panel.parquet"

TOP_COUNTRIES = 6
MIN_DAYS_PER_COUNTRY = 200


KEYWORD_CATEGORIES = [
    ("Home & Kitchen",  ["BOWL", "PLATE", "MUG", "CUP", "JUG", "TRAY", "COASTER", "KITCHEN"]),
    ("Decor",           ["HEART", "CANDLE", "LANTERN", "GARLAND", "DECORATION", "FRAME"]),
    ("Gifts & Cards",   ["CARD", "GIFT", "WRAP", "RIBBON", "TAG"]),
    ("Bags & Storage",  ["BAG", "LUNCH", "BASKET", "BOX", "TIN"]),
    ("Stationery",      ["PENCIL", "PEN", "NOTEBOOK", "PAPER", "ERASER", "RUBBER", "STICKER"]),
]


def _assign_category(desc: str) -> str:
    text = (desc or "").upper()
    for name, kws in KEYWORD_CATEGORIES:
        if any(kw in text for kw in kws):
            return name
    return "Other"


def download_raw(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_XLSX.exists() and not force:
        return RAW_XLSX
    req = Request(URL, headers={"User-Agent": "python-urllib/3"})
    with urlopen(req, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".xlsx"):
                with zf.open(name) as src, open(RAW_XLSX, "wb") as dst:
                    dst.write(src.read())
                break
    return RAW_XLSX


def _load_raw() -> pd.DataFrame:
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None, engine="openpyxl")
    frames = []
    for name, sheet in sheets.items():
        sheet = sheet.rename(columns={
            "Invoice": "invoice", "StockCode": "stock_code",
            "Description": "description", "Quantity": "quantity",
            "InvoiceDate": "invoice_date", "Price": "price",
            "Customer ID": "customer_id", "Country": "country",
        })
        sheet["sheet"] = name
        frames.append(sheet)
    return pd.concat(frames, ignore_index=True)


def to_panel(save: bool = True, force: bool = False) -> pd.DataFrame:
    download_raw(force=force)
    tx = _load_raw()

    tx = tx.dropna(subset=["invoice_date", "quantity", "price", "country"])
    tx = tx[(tx["quantity"] > 0) & (tx["price"] > 0)]
    tx = tx[~tx["invoice"].astype(str).str.startswith("C", na=False)]
    tx["revenue"] = tx["quantity"] * tx["price"]
    tx["date"] = pd.to_datetime(tx["invoice_date"]).dt.normalize()
    tx["category"] = tx["description"].astype(str).map(_assign_category)

    country_days = tx.groupby("country")["date"].nunique().sort_values(ascending=False)
    keep = country_days[country_days >= MIN_DAYS_PER_COUNTRY].head(TOP_COUNTRIES).index.tolist()
    tx = tx[tx["country"].isin(keep)].copy()
    tx = tx[tx["category"] != "Other"]

    panel = (
        tx.groupby(["date", "category", "country"], observed=True)
        .agg(revenue=("revenue", "sum"), units_sold=("quantity", "sum"))
        .reset_index()
        .rename(columns={"country": "region"})
    )

    panel = panel[panel["revenue"] > 0].copy()
    panel["avg_price"] = (panel["revenue"] / panel["units_sold"]).round(2)

    full_index = pd.MultiIndex.from_product(
        [
            pd.date_range(panel["date"].min(), panel["date"].max(), freq="D"),
            panel["category"].unique(),
            panel["region"].unique(),
        ],
        names=["date", "category", "region"],
    )
    panel = panel.set_index(["date", "category", "region"]).reindex(full_index).reset_index()
    panel = panel.sort_values(["category", "region", "date"]).reset_index(drop=True)

    for col in ["revenue", "units_sold", "avg_price"]:
        panel[col] = panel.groupby(["category", "region"], observed=True)[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )

    cogs_by_category = {
        "Home & Kitchen": 0.55, "Decor": 0.48, "Gifts & Cards": 0.52,
        "Bags & Storage": 0.58, "Stationery": 0.50,
    }
    panel["cogs"] = (panel["revenue"] * panel["category"].map(cogs_by_category)).round(2)
    panel["gross_profit"] = (panel["revenue"] - panel["cogs"]).round(2)
    panel = panel.dropna(subset=["revenue", "units_sold"]).reset_index(drop=True)

    if save:
        PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(PANEL_PATH, index=False)
    return panel


def load(force_rebuild: bool = False) -> pd.DataFrame:
    if PANEL_PATH.exists() and not force_rebuild:
        return pd.read_parquet(PANEL_PATH)
    return to_panel(save=True, force=force_rebuild)
