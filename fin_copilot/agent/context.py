from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from fin_copilot.config import ARTIFACTS, load_config
from fin_copilot.data.loader import load
from fin_copilot.forecasting.model import QuantileForecaster

MODEL_PATH = ARTIFACTS / "forecaster.pkl"


@dataclass
class CopilotContext:
    panel: pd.DataFrame
    model: QuantileForecaster
    categories: list[str]
    regions: list[str]


def _train_and_cache(panel: pd.DataFrame) -> QuantileForecaster:
    cfg = load_config()
    model = QuantileForecaster(quantiles=cfg.forecast.quantiles, params=cfg.forecast.lgbm)
    model.fit(panel)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(model, fh)
    return model


def _load_or_train(panel: pd.DataFrame, force_retrain: bool = False) -> QuantileForecaster:
    if MODEL_PATH.exists() and not force_retrain:
        with open(MODEL_PATH, "rb") as fh:
            return pickle.load(fh)
    return _train_and_cache(panel)


@lru_cache(maxsize=1)
def get_context(force_retrain: bool = False) -> CopilotContext:
    panel = load()
    model = _load_or_train(panel, force_retrain=force_retrain)
    return CopilotContext(
        panel=panel,
        model=model,
        categories=sorted(panel["category"].unique().tolist()),
        regions=sorted(panel["region"].unique().tolist()),
    )


def reset_context() -> None:
    get_context.cache_clear()


def override_context(panel: pd.DataFrame, model: QuantileForecaster) -> CopilotContext:
    ctx = CopilotContext(
        panel=panel,
        model=model,
        categories=sorted(panel["category"].unique().tolist()),
        regions=sorted(panel["region"].unique().tolist()),
    )
    get_context.cache_clear()
    _OVERRIDE["ctx"] = ctx
    return ctx


_OVERRIDE: dict[str, CopilotContext | None] = {"ctx": None}


def current_context() -> CopilotContext:
    if _OVERRIDE["ctx"] is not None:
        return _OVERRIDE["ctx"]
    return get_context()


def clear_override() -> None:
    _OVERRIDE["ctx"] = None
