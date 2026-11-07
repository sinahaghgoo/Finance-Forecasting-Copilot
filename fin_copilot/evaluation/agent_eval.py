from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_\.])-?\d{1,3}(?:[,_]\d{3})*(?:\.\d+)?(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])"
)
TRIVIAL_NUMBERS = {0.0, 1.0, 10.0, 100.0, 1000.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}


def extract_numbers(text: str) -> list[float]:
    found = []
    for m in NUMBER_RE.finditer(text or ""):
        tok = m.group(0).replace(",", "").replace("_", "")
        try:
            found.append(float(tok))
        except ValueError:
            continue
    return found


def _flatten_numbers(value: Any) -> list[float]:
    out: list[float] = []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append(float(value))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_flatten_numbers(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(_flatten_numbers(v))
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                out.extend(_flatten_numbers(parsed))
                return out
        except (json.JSONDecodeError, ValueError):
            pass
        out.extend(extract_numbers(value))
    return out


def numbers_from_tool_results(tool_results: list[dict]) -> list[float]:
    all_numbers: list[float] = []
    for tr in tool_results:
        all_numbers.extend(_flatten_numbers(tr.get("content")))
    return all_numbers


def _is_grounded(value: float, pool: list[float], rel_tol: float = 0.02, abs_tol: float = 0.5) -> bool:
    if value in TRIVIAL_NUMBERS or abs(value) < 1e-6:
        return True
    for ref in pool:
        if ref in TRIVIAL_NUMBERS:
            continue
        if abs(value - ref) <= max(abs_tol, abs(ref) * rel_tol):
            return True
        if abs(value) <= 100 and abs(ref) > 1 and abs(value - ref * 100) <= max(abs_tol, abs(ref) * rel_tol * 100):
            return True
    return False


@dataclass
class GroundingReport:
    answer_numbers: list[float]
    tool_numbers: list[float]
    grounded: list[float]
    hallucinated: list[float]

    @property
    def hallucination_rate(self) -> float:
        meaningful = [n for n in self.answer_numbers if n not in TRIVIAL_NUMBERS]
        if not meaningful:
            return 0.0
        return len(self.hallucinated) / len(meaningful)


def score_grounding(answer: str, tool_results: list[dict]) -> GroundingReport:
    ans_nums = extract_numbers(answer)
    pool = numbers_from_tool_results(tool_results)
    grounded, hallucinated = [], []
    for n in ans_nums:
        if _is_grounded(n, pool):
            grounded.append(n)
        else:
            hallucinated.append(n)
    return GroundingReport(ans_nums, pool, grounded, hallucinated)


@dataclass
class AgentCase:
    question: str
    expected_answer_near: float | None = None
    expected_tools: list[str] = field(default_factory=list)
    tolerance_pct: float = 5.0

    def accuracy_ok(self, answer_numbers: list[float]) -> bool | None:
        if self.expected_answer_near is None:
            return None
        if not answer_numbers:
            return False
        target = self.expected_answer_near
        tol = max(abs(target) * self.tolerance_pct / 100, 1e-6)
        return any(abs(n - target) <= tol for n in answer_numbers)

    def tool_coverage(self, tool_calls: list[dict]) -> float:
        if not self.expected_tools:
            return 1.0
        names = {tc["name"] for tc in tool_calls}
        return sum(1 for t in self.expected_tools if t in names) / len(self.expected_tools)


@dataclass
class EvaluationRow:
    question: str
    answer: str
    tool_calls: list[dict]
    hallucination_rate: float
    accuracy_ok: bool | None
    tool_coverage: float


def evaluate_agent(agent_invoker: Callable[[str], dict], cases: list[AgentCase]) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    for case in cases:
        result = agent_invoker(case.question)
        grounding = score_grounding(result["answer"], result["tool_results"])
        rows.append(EvaluationRow(
            question=case.question,
            answer=result["answer"],
            tool_calls=result["tool_calls"],
            hallucination_rate=grounding.hallucination_rate,
            accuracy_ok=case.accuracy_ok(grounding.answer_numbers),
            tool_coverage=case.tool_coverage(result["tool_calls"]),
        ))
    return rows


def aggregate(rows: list[EvaluationRow]) -> dict:
    n = len(rows)
    acc_rows = [r for r in rows if r.accuracy_ok is not None]
    acc = sum(1 for r in acc_rows if r.accuracy_ok) / len(acc_rows) if acc_rows else None
    hall = sum(r.hallucination_rate for r in rows) / n if n else 0.0
    cov = sum(r.tool_coverage for r in rows) / n if n else 0.0
    return {
        "n_cases": n,
        "accuracy": acc,
        "mean_hallucination_rate": hall,
        "mean_tool_coverage": cov,
    }
