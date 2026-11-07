from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"


@dataclass
class DataConfig:
    start_date: str
    end_date: str
    categories: list[str]
    regions: list[str]
    seed: int


@dataclass
class LGBMParams:
    num_leaves: int = 64
    learning_rate: float = 0.05
    n_estimators: int = 400
    min_data_in_leaf: int = 30
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    verbose: int = -1


@dataclass
class ForecastConfig:
    horizon_days: int
    quantiles: list[float]
    lgbm: LGBMParams


@dataclass
class BacktestConfig:
    n_folds: int
    fold_horizon_days: int


@dataclass
class AgentConfig:
    model_provider: str
    model_name: str
    temperature: float = 0.0


@dataclass
class Config:
    data: DataConfig
    forecast: ForecastConfig
    backtest: BacktestConfig
    agent: AgentConfig = field(default_factory=lambda: AgentConfig("openai", "gpt-4o-mini"))


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return Config(
        data=DataConfig(**raw["data"]),
        forecast=ForecastConfig(
            horizon_days=raw["forecast"]["horizon_days"],
            quantiles=raw["forecast"]["quantiles"],
            lgbm=LGBMParams(**raw["forecast"]["lgbm"]),
        ),
        backtest=BacktestConfig(**raw["backtest"]),
        agent=AgentConfig(**raw["agent"]),
    )
