from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "revenue"
GROUP_COLS = ["category", "region"]
LAGS = [7, 14, 28, 56, 365]
ROLLING_WINDOWS = [7, 28]


def add_calendar(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = df[date_col]
    df = df.copy()
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dow"] = d.dt.dayofweek
    df["doy"] = d.dt.dayofyear
    df["week"] = d.dt.isocalendar().week.astype(int)
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    df["sin_doy"] = np.sin(2 * np.pi * d.dt.dayofyear / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * d.dt.dayofyear / 365.25)
    return df


def add_lags(df: pd.DataFrame, target: str = TARGET, lags: list[int] = LAGS) -> pd.DataFrame:
    df = df.sort_values(GROUP_COLS + ["date"]).copy()
    g = df.groupby(GROUP_COLS, observed=True)[target]
    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag)
    return df


def add_rolling(df: pd.DataFrame, target: str = TARGET, windows: list[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    df = df.sort_values(GROUP_COLS + ["date"]).copy().reset_index(drop=True)
    g = df.groupby(GROUP_COLS, observed=True)[target]
    for w in windows:
        df[f"roll_mean_{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).mean())
        df[f"roll_std_{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).std())
    return df


def encode_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in GROUP_COLS:
        df[col] = df[col].astype("category")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_calendar(df)
    out = add_lags(out)
    out = add_rolling(out)
    out = encode_groups(out)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"date", TARGET, "units_sold", "avg_price", "cogs", "gross_profit"}
    return [c for c in df.columns if c not in drop]
