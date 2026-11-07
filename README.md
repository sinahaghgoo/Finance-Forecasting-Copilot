# Finance Forecasting Copilot

A small, end-to-end analytics system that a finance team could actually sit next to:

- **Forecasts** daily revenue per product category and region with a LightGBM quantile
  model, validated by walk-forward backtesting.
- **Projects** gross profit under what-if scenarios on price, cost and demand elasticity.
- **Answers questions in natural language** through a LangGraph agent that is only allowed
  to reply with numbers produced by its own tools, and that is measured on accuracy and
  hallucination rate.

The pipeline is designed to be data-source-agnostic. It ships with three loaders behind the
same panel schema: a **synthetic** generator baked into the repo (the default, so everything
is reproducible with no external dependencies), **UCI Online Retail II** (a public UK-based
online-retail dataset, downloaded directly from UCI), and **Kaggle Rossmann Store Sales**.
The forecaster, scenario engine and agent run unchanged on any of the three.

### Preview on GitHub (no install needed)

- [notebooks/01_walkthrough.ipynb](notebooks/01_walkthrough.ipynb) — synthetic data, executed end to end: forecast plot + intervals, 6-fold backtest, scenario comparison, and the **LangGraph agent answering three real questions** via a local Ollama model.
- [notebooks/02_real_data_walkthrough.ipynb](notebooks/02_real_data_walkthrough.ipynb) — same pipeline on UCI Online Retail II, executed on the UK subset.

---

## Why this project

Two things are hard about analytics in finance:

1. **Trustworthy forecasts** – headline point forecasts hide the uncertainty that finance
   decisions depend on. Any serious forecaster needs intervals and honest backtesting.
2. **Trustworthy chat** – LLMs will cheerfully invent a number. A finance copilot is only
   useful if every number it says can be traced to a verified source.

The project takes both seriously. The forecast layer reports 80% prediction intervals and
is scored on quantile loss and empirical coverage. The agent layer is evaluated on a
hallucination rate that flags any number in the final answer which does not come from a
tool output.

---

## Architecture

```
          ┌──────────────────────────────────────────────┐
          │  Pluggable data source                       │
          │   • synthetic panel (default, reproducible)  │
          │   • UCI Online Retail II                     │
          │   • Kaggle Rossmann Store Sales              │
          │  → normalised to one (date, category,        │
          │     region, revenue, units, price, cogs, gp) │
          └────────────────────┬─────────────────────────┘
                               │
      feature engineering ▾    (calendar, lags, rolling stats)
                       │
          ┌────────────▼─────────────┐
          │  LightGBM quantile model │   ← one booster per quantile (q10/q50/q90)
          │  recursive multi-step    │     post-sorted for monotonicity
          └──────┬─────────────┬─────┘
                 │             │
    walk-forward │             │ point & interval forecast
      backtest   │             │
    (MAPE, RMSE, │             ▼
    pinball,     │   ┌───────────────────────────┐
    coverage)    │   │  Gross-profit scenario    │
                 │   │  engine (price × cost ×   │
                 │   │  elasticity → revenue,    │
                 │   │  cogs, gross profit)      │
                 │   └────────────┬──────────────┘
                 │                │
                 ▼                ▼
          ┌──────────────────────────────┐
          │  LangGraph agent             │
          │  tools:                      │
          │   • list_dimensions          │
          │   • historical_metric        │
          │   • forecast_revenue         │
          │   • gross_profit_scenario    │
          └────────────┬─────────────────┘
                       │
             ┌─────────▼─────────┐
             │ Grounding check   │  numbers in answer must appear
             │ • accuracy        │  in tool outputs → otherwise flagged
             │ • hallucination % │
             │ • tool coverage   │
             └───────────────────┘
```

---

## Results

### Synthetic panel (default and fully reproduceable)

Walk-forward backtest, 6 folds × 28-day horizon, held out from the tail:

| Metric           | Mean   | Std    |
|------------------|--------|--------|
| MAPE             | 5.01 % | 0.86   |
| sMAPE            | 4.96 % | 0.90   |
| RMSE             | 24,124 | 9,369  |
| Pinball @ q10    | 3,815  | 894    |
| Pinball @ q90    | 4,290  | 1,726  |
| 80% PI coverage  | 0.65   | 0.04   |

### Rossmann Store Sales (real data, Kaggle CC0 mirror)

Same model and features, 4 folds × 28-day horizon on 6 (store-type × assortment) series
built from ~1.6 M daily store records:

| Metric           | Mean   |
|------------------|--------|
| MAPE             | ~ 11 % |
| sMAPE            | ~ 11 % |
| 80% PI coverage  | ~ 0.52 |

A clean mid-single-digit-to-low-double-digit error on 28-day holdouts of real multi-store
retail sales. The loader filters out days where fewer than half the usual stores were open
(German public holidays, mostly) and drops storetype × assortment combinations with fewer
than 30 stores, since tiny-sample series behave more like noise than signal.

### UCI Online Retail II (real data, UK subset)

Same model, 4 folds × 14-day horizon on the UK series only (the countries with volume high
enough for percentage-based metrics to be meaningful):

| Metric              | Approx. |
|---------------------|---------|
| MAPE (per category) | 30 – 50 % |
| 80% PI coverage     | ~ 0.63 |

Online Retail II is noticeably harder than Rossmann — much shorter history (~740 days vs
913), only one prior Christmas peak to learn from, and no promo-calendar features. The
point forecast lands within tens of percent, but the 80% intervals still catch roughly
two-thirds of the outcomes. That is the honest value of the uncertainty layer: when the
point estimate is weak the model says so, instead of being confidently wrong.

Natural next steps for either real dataset: weekly aggregation for the planning view,
exogenous promo/marketing features, and isotonic or conformal calibration of the quantiles.

---

## Data sources

| Source | License | Link |
|---|---|---|
| Synthetic finance panel (3 categories × 3 regions × 5 years, daily) | MIT, generated in-repo by `fin_copilot/data/generate.py` | — |
| UCI Online Retail II | CC BY 4.0 (UCI Machine Learning Repository) | <https://archive.ics.uci.edu/dataset/502/online+retail+ii> |
| Kaggle Rossmann Store Sales (CC0 mirror) | CC0 1.0 Public Domain | <https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales> |
| Original Rossmann competition (reference, not downloaded) | Kaggle competition T&Cs | <https://www.kaggle.com/competitions/rossmann-store-sales> |

The Rossmann mirror exposes the same `train.csv` and `store.csv` as the original competition
but is published as a plain CC0 dataset, so no competition-rules acceptance is needed — any
Kaggle account that can run `kaggle datasets list` can download it.

---

## Getting started

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Build the data and train the forecaster

```bash
python scripts/train_and_backtest.py
```

This generates the synthetic panel (`data/raw/finance_panel.parquet`), runs the walk-forward
backtest, writes metrics to `artifacts/`, and caches the trained model.

### Switching data source

The active data source is chosen by the `FINCOPILOT_DATA_SOURCE` environment variable:

```bash
# synthetic (default) — no setup needed
fincopilot backtest

# UCI Online Retail II — downloads ~45 MB from archive.ics.uci.edu on first use
FINCOPILOT_DATA_SOURCE=online_retail fincopilot backtest

# Kaggle Rossmann Store Sales — CC0 mirror, authenticated download
pip install -e ".[kaggle]"
python scripts/fetch_rossmann.py           # downloads ~9 MB and builds the panel (mirrored the data like that so Kaggle Auth. is no problem)
FINCOPILOT_DATA_SOURCE=rossmann fincopilot backtest
```

The Rossmann loader pulls from the CC0 mirror
`pratyushakar/rossmann-store-sales` (same `train.csv` / `store.csv` as the original
competition) so no rules acceptance is needed. Any Kaggle account that can run
`kaggle datasets list` is enough.

### CLI

```bash
fincopilot build-data                              # regenerate the synthetic panel
fincopilot backtest --folds 6 --horizon 28         # walk-forward CV
fincopilot forecast --category Footwear --region EMEA --horizon 14
fincopilot ask "Forecast Footwear revenue in EMEA over the next 14 days"
fincopilot eval-agent                              # full agent evaluation
```

The `ask` and `eval-agent` commands need an LLM backend. The default is **Ollama** running
locally — free, no API key, works offline:

```bash
brew install ollama              # or: https://ollama.com
brew services start ollama       # starts the local daemon on :11434
ollama pull qwen2.5:7b           # ~4.7 GB; good tool-calling quality
pip install -e ".[ollama]"
fincopilot ask "Forecast Footwear revenue in EMEA over the next 14 days"
```

Hosted providers are supported too — set `FINCOPILOT_MODEL_PROVIDER=openai` or `anthropic`
and the matching `*_API_KEY` in `.env` (see `.env.example`). All other commands run offline
and need neither Ollama nor an API key.

### Docker

```bash
docker build -t fin-copilot .
docker run --rm fin-copilot backtest --folds 3 --horizon 14
```

### Tests

```bash
pytest -q
ruff check fin_copilot tests scripts
```

The 55-test suite covers the synthetic generator, both real-data loaders (with mocked
downloads), feature engineering, the LightGBM forecaster, the scenario engine, the
grounding metric, CLI dispatch, and the agent graph driven by a scripted chat model —
so neither Ollama nor an API key is needed to run the tests.

---

## Repository layout

```
fin_copilot/
├── data/
│   ├── generate.py       synthetic finance panel generator
│   ├── rossmann.py       Kaggle Rossmann loader (CC0 mirror)
│   ├── online_retail.py  UCI Online Retail II loader
│   └── loader.py         dispatches to the active source (FINCOPILOT_DATA_SOURCE)
├── forecasting/          features, LightGBM quantile model, walk-forward backtest
├── scenarios/            gross-profit what-if engine (price × cost × elasticity)
├── agent/                LangGraph agent, tools, prompt, cached model/context
│                         providers: Ollama (default, local), OpenAI, Anthropic
└── evaluation/           forecast metrics, grounding / hallucination scoring, eval cases

tests/                    55-test pytest suite covering data generators, loaders
                          (with mocked downloads), features, LightGBM, scenarios,
                          grounding, CLI dispatch, and agent graph (scripted LLM)
scripts/                  train_and_backtest.py, fetch_rossmann.py
notebooks/
├── 01_walkthrough.ipynb             synthetic panel · data → forecast → scenario
│                                    → agent (runs against local Ollama, outputs
│                                    committed so GitHub shows real tool traces)
└── 02_real_data_walkthrough.ipynb   UCI Online Retail II · same pipeline, real data
configs/default.yaml      data, forecast, backtest, and agent (Ollama by default)
```

---

## How the agent stays grounded

The agent is a plain LangGraph tool-calling loop (`agent → tools → agent → …`) but the
evaluation layer treats every number in the final answer as a claim that must be traced
back to a tool call:

1. The system prompt tells the model that every numeric value in its reply must come from
   a tool result, and never from its own knowledge.
2. During evaluation, `score_grounding` extracts all numbers from the final answer and
   from the recorded tool outputs. A number in the answer is marked **grounded** if it
   matches a value anywhere in the tool outputs within a small tolerance, and
   **hallucinated** otherwise.
3. The mean hallucination rate across a fixed set of questions becomes a KPI that would
   gate a release if it went above a threshold.

This is a deliberately simple grounding check – the point isn't to catch every subtle
misquote, it's to have a real guardrail that would surface a model that drifts off its
tool outputs. A production version would layer on things like structured JSON responses
with explicit source IDs per number, and a human-in-the-loop review for edge cases.

---

## What I'd add next

- **More features** – known promotional calendars, exchange-rate signals, inventory
  constraints. Right now the model infers promos from history alone.
- **More tools** – a `compare_scenarios` tool that runs two scenarios side by side and
  returns the diff, a `forecast_gross_profit` tool that bakes assumed cost paths in.
- **Evaluation harness** – expand to a few dozen seeded questions, split by
  question type (lookup / forecast / scenario / out-of-scope) and track hallucination
  rate by type.
- **Interval calibration** – isotonic or split-conformal calibration on a held-out window
  to close the coverage gap.
- **Deployment** – the Dockerfile builds a thin runtime image; a real deployment would
  split model training (batch job) from the agent server (FastAPI) and persist the
  trained model somewhere durable.

---

## Stack

| Area              | Tools                                                  |
|-------------------|--------------------------------------------------------|
| Data & analytics  | Python, pandas, numpy, SQL-style groupby, statsmodels  |
| Forecasting       | LightGBM (quantile objective), scikit-learn conventions |
| Agent             | LangGraph, LangChain; Ollama (default, local), OpenAI, Anthropic |
| Evaluation        | pytest, scripted LangChain fake-model for agent tests  |
| Packaging         | setuptools / pyproject, Docker, ruff                   |

---

