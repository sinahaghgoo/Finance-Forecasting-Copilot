from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from fin_copilot.agent import context as ctx_mod
from fin_copilot.agent.graph import build_agent
from fin_copilot.agent.tools import (
    forecast_revenue,
    gross_profit_scenario,
    historical_metric,
    list_dimensions,
)
from fin_copilot.config import LGBMParams
from fin_copilot.data.generate import generate_panel
from fin_copilot.forecasting.model import QuantileForecaster


@pytest.fixture(scope="module")
def small_context():
    panel = generate_panel("2023-01-01", "2024-02-28", ["Footwear", "Apparel"], ["EMEA", "APAC"], seed=5)
    params = LGBMParams(n_estimators=100, num_leaves=31, min_data_in_leaf=20)
    model = QuantileForecaster(quantiles=[0.1, 0.5, 0.9], params=params).fit(panel)
    ctx = ctx_mod.override_context(panel, model)
    yield ctx
    ctx_mod.clear_override()


def test_list_dimensions_tool(small_context):
    out = list_dimensions.invoke({})
    assert out["categories"] == ["Apparel", "Footwear"]
    assert out["regions"] == ["APAC", "EMEA"]
    assert out["data_start"] == "2023-01-01"


def test_historical_metric_returns_expected_total(small_context):
    out = historical_metric.invoke({
        "category": "Footwear", "region": "EMEA",
        "start_date": "2023-05-01", "end_date": "2023-05-31",
        "metric": "revenue",
    })
    panel = small_context.panel
    mask = (
        (panel["category"] == "Footwear")
        & (panel["region"] == "EMEA")
        & (panel["date"] >= "2023-05-01")
        & (panel["date"] <= "2023-05-31")
    )
    expected = float(panel.loc[mask, "revenue"].sum())
    assert abs(out["total"] - expected) < 1e-6


def test_historical_metric_rejects_bad_metric(small_context):
    with pytest.raises(ValueError, match="Unknown metric"):
        historical_metric.invoke({
            "category": "Footwear", "region": "EMEA",
            "start_date": "2023-05-01", "end_date": "2023-05-31",
            "metric": "profit_margin",
        })


def test_historical_metric_rejects_unknown_category(small_context):
    with pytest.raises(ValueError, match="Unknown category"):
        historical_metric.invoke({
            "category": "Accessories", "region": "EMEA",
            "start_date": "2023-05-01", "end_date": "2023-05-31",
        })


def test_forecast_revenue_tool(small_context):
    out = forecast_revenue.invoke({"category": "Footwear", "region": "EMEA", "horizon_days": 7})
    assert out["horizon_days"] == 7
    assert out["total_q10"] <= out["total_point"] <= out["total_q90"]
    assert len(out["daily"]) == 7


def test_gross_profit_scenario_tool(small_context):
    out = gross_profit_scenario.invoke({
        "category": "Footwear", "region": "EMEA",
        "horizon_days": 14, "price_change_pct": 0.05, "price_elasticity": -1.0,
    })
    assert "scenario_gross_profit" in out
    assert "baseline_gross_profit" in out
    assert out["scenario_gross_profit"] > out["baseline_gross_profit"]


class ScriptedChatModel(BaseChatModel):
    """Minimal chat model that plays back a scripted sequence of AIMessage replies."""

    responses: list
    cursor: dict = {"i": 0}

    def __init__(self, responses: list[AIMessage]):
        super().__init__(responses=responses, cursor={"i": 0})

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools):
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs) -> ChatResult:
        i = self.cursor["i"]
        reply = self.responses[i]
        self.cursor["i"] = i + 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def test_agent_graph_runs_scripted_tool_call(small_context):
    scripted = [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "call_1",
                "name": "list_dimensions",
                "args": {},
            }],
        ),
        AIMessage(content="We have Footwear and Apparel in EMEA and APAC."),
    ]
    agent = build_agent(llm=ScriptedChatModel(scripted))
    result = agent.invoke("What data do you have?")
    assert "Footwear" in result["answer"]
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "list_dimensions"
    assert len(result["tool_results"]) == 1


def test_agent_grounding_catches_fabricated_number(small_context):
    scripted = [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "call_1",
                "name": "historical_metric",
                "args": {
                    "category": "Footwear", "region": "EMEA",
                    "start_date": "2023-05-01", "end_date": "2023-05-31", "metric": "revenue",
                },
            }],
        ),
        AIMessage(content="Revenue was EUR 999,999,999 in May 2023."),
    ]
    from fin_copilot.evaluation.agent_eval import score_grounding
    agent = build_agent(llm=ScriptedChatModel(scripted))
    result = agent.invoke("What was May 2023 Footwear revenue in EMEA?")
    report = score_grounding(result["answer"], result["tool_results"])
    assert report.hallucination_rate > 0
