from __future__ import annotations

from fin_copilot.evaluation.agent_eval import (
    AgentCase,
    extract_numbers,
    numbers_from_tool_results,
    score_grounding,
)


def test_extract_numbers_handles_commas_and_decimals():
    txt = "Revenue was 12,345,678.90 EUR and growth is 3.2%."
    nums = extract_numbers(txt)
    assert 12345678.90 in nums
    assert 3.2 in nums


def test_numbers_from_tool_results_recurses_into_dict():
    tr = [{"id": "t1", "name": "x", "content": {"total": 1234.5, "nested": [{"v": 10.0}]}}]
    nums = numbers_from_tool_results(tr)
    assert 1234.5 in nums
    assert 10.0 in nums


def test_score_grounding_marks_matching_numbers_as_grounded():
    tool_results = [{"id": "t1", "name": "h", "content": {"total": 12345.67}}]
    answer = "Total was 12,345.67 EUR"
    rep = score_grounding(answer, tool_results)
    assert 12345.67 in rep.grounded
    assert not rep.hallucinated


def test_score_grounding_catches_hallucination():
    tool_results = [{"id": "t1", "name": "h", "content": {"total": 1000.0}}]
    answer = "Total was 9999.0 EUR"
    rep = score_grounding(answer, tool_results)
    assert 9999.0 in rep.hallucinated
    assert rep.hallucination_rate > 0


def test_agent_case_accuracy_tolerance():
    c = AgentCase(question="q", expected_answer_near=100.0, tolerance_pct=5.0)
    assert c.accuracy_ok([104.0]) is True
    assert c.accuracy_ok([106.0]) is False
    assert c.accuracy_ok([]) is False
