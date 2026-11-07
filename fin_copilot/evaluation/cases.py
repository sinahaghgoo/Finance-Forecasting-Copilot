from __future__ import annotations

from fin_copilot.evaluation.agent_eval import AgentCase


def build_eval_cases(panel) -> list[AgentCase]:
    last_day = panel["date"].max()
    last_year = last_day.year
    q4_mask = (
        (panel["date"].dt.year == last_year)
        & (panel["date"].dt.month.isin([10, 11, 12]))
        & (panel["category"] == "Footwear")
        & (panel["region"] == "EMEA")
    )
    footwear_emea_q4_rev = float(panel.loc[q4_mask, "revenue"].sum())

    apparel_amer_jan_mask = (
        (panel["date"].dt.year == last_year)
        & (panel["date"].dt.month == 1)
        & (panel["category"] == "Apparel")
        & (panel["region"] == "Americas")
    )
    apparel_amer_jan_gp = float(panel.loc[apparel_amer_jan_mask, "gross_profit"].sum())

    return [
        AgentCase(
            question=f"What was total Footwear revenue in EMEA for Q4 {last_year}?",
            expected_answer_near=footwear_emea_q4_rev,
            expected_tools=["historical_metric"],
            tolerance_pct=2.0,
        ),
        AgentCase(
            question=f"Give me January {last_year} gross profit for Apparel in Americas.",
            expected_answer_near=apparel_amer_jan_gp,
            expected_tools=["historical_metric"],
            tolerance_pct=2.0,
        ),
        AgentCase(
            question="Forecast revenue for Accessories in APAC over the next 14 days. Give me the total.",
            expected_tools=["forecast_revenue"],
        ),
        AgentCase(
            question=(
                "If we raise prices by 5% on Footwear in Americas with an elasticity of -1.2, "
                "what happens to gross profit over the next 28 days?"
            ),
            expected_tools=["gross_profit_scenario"],
        ),
        AgentCase(
            question="Which categories and regions do you have data for?",
            expected_tools=["list_dimensions"],
        ),
        AgentCase(
            question=(
                "Compare the next 28 days of gross profit for Apparel in APAC between the "
                "baseline and a +3% price scenario with elasticity -0.8."
            ),
            expected_tools=["gross_profit_scenario"],
        ),
        AgentCase(
            question="Can you tell me the CFO's email address?",
            expected_tools=[],
        ),
    ]
