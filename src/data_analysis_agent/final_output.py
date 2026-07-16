"""Bounded, formatting-only final-output generation and repair helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import JsonValue

from data_analysis_agent.schemas import FinalAnswer
from data_analysis_agent.state import IterationRecord


@dataclass(frozen=True)
class FinalGenerationRequest:
    """The deliberately narrow context supplied to final-answer generation."""

    question: str
    approved_execution_result: str
    verifier_decision: str
    verifier_feedback: str
    iteration_history: list[IterationRecord]
    completed_goal_results: list[dict[str, JsonValue]] = field(default_factory=list)
    answer_schema: dict[str, JsonValue] | None = None
    final_output_goal_id: str | None = None


@dataclass(frozen=True)
class OutputRepairRequest:
    """The only inputs available to output repair."""

    invalid_raw_output: str
    validation_error: str
    required_schema: dict[str, object]
    approved_execution_result: str


class FinalOutputProvider(Protocol):
    """Formatting boundary used by deterministic and scripted providers."""

    def generate(self, request: FinalGenerationRequest) -> str:
        """Return one candidate JSON string from an approved execution result."""
        ...

    def repair(self, request: OutputRepairRequest) -> str:
        """Repair only syntax/schema issues in a candidate output."""
        ...


_RESULT_PATTERNS = {
    "mean": re.compile(r"\bmean\s*(?:=|≈|:)\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    "sample_standard_error": re.compile(
        r"\b(?:sample\s+)?standard error\s*(?:=|≈|:)\s*"
        r"(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    "n_observations": re.compile(
        r"\b(?:number of observations used|observations|n_observations)\s*"
        r"(?:=|≈|:)\s*(\d+)",
        re.IGNORECASE,
    ),
}


def _extract_key_results(approved_execution_result: str) -> dict[str, int | float]:
    """Extract only explicitly labeled values without recalculating them."""
    results: dict[str, int | float] = {}
    for name, pattern in _RESULT_PATTERNS.items():
        match = pattern.search(approved_execution_result)
        if match is None:
            continue
        value = match.group(1)
        results[name] = int(value) if name == "n_observations" else float(value)
    return results


def _approved_structured_values(
    completed_goal_results: list[dict[str, JsonValue]],
) -> tuple[dict[str, JsonValue], list[str]]:
    """Copy numeric/script values from approved results without recalculation."""
    key_results: dict[str, JsonValue] = {}
    limitations: list[str] = []
    for goal_result in completed_goal_results:
        if not goal_result.get("success"):
            continue
        warnings = goal_result.get("warnings", [])
        if isinstance(warnings, list):
            limitations.extend(str(item) for item in warnings)
        result = goal_result.get("result", {})
        if not isinstance(result, dict):
            continue
        if goal_result.get("capability_name") == "compute_summary_statistics":
            statistics_result = result.get("statistics", {})
            if isinstance(statistics_result, dict):
                for name, value in statistics_result.items():
                    output_name = "n_observations" if name == "count" else name
                    key_results[output_name] = value
    generated = [
        item
        for item in completed_goal_results
        if item.get("success") and item.get("strategy") == "generated_python"
    ]
    if generated:
        result = generated[-1].get("result", {})
        if isinstance(result, dict) and "legacy_execution_result" not in result:
            key_results = dict(result)
    return key_results, list(dict.fromkeys(limitations))


def _serialize_approved_result(
    approved_execution_result: str,
    completed_goal_results: list[dict[str, JsonValue]] | None = None,
) -> str:
    goal_count = len(completed_goal_results or [])
    structured_results, limitations = _approved_structured_values(
        completed_goal_results or []
    )
    if not structured_results and not completed_goal_results:
        try:
            approved_object = json.loads(approved_execution_result)
        except json.JSONDecodeError:
            approved_object = None
        if isinstance(approved_object, dict) and isinstance(
            approved_object.get("completed_goal_results"), list
        ):
            goal_count = len(approved_object["completed_goal_results"])
            structured_results, limitations = _approved_structured_values(
                approved_object["completed_goal_results"]
            )
    if structured_results:
        answer_text = (
            f"Completed {goal_count} verified goal(s). "
            f"Key results: {json.dumps(structured_results, ensure_ascii=False)}"
        )
        key_results = structured_results
    else:
        answer_text = approved_execution_result.strip()
        key_results = _extract_key_results(approved_execution_result)
    answer = FinalAnswer(
        status="completed",
        answer=answer_text,
        key_results=key_results,
        limitations=limitations,
    )
    return json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2)


class DeterministicFinalOutputProvider:
    """Format approved content without asking a model to judge or recalculate it."""

    def generate(self, request: FinalGenerationRequest) -> str:
        if request.answer_schema is not None:
            matches = [
                item
                for item in request.completed_goal_results
                if item.get("goal_id") == request.final_output_goal_id
                and item.get("success")
            ]
            if len(matches) != 1:
                return json.dumps(
                    {"assembly_error": "verified final assembly result is missing"}
                )
            return json.dumps(
                matches[0].get("result", {}),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        return _serialize_approved_result(
            request.approved_execution_result,
            request.completed_goal_results,
        )

    def repair(self, request: OutputRepairRequest) -> str:
        try:
            parsed = json.loads(request.approved_execution_result)
        except json.JSONDecodeError:
            return _serialize_approved_result(request.approved_execution_result)
        if isinstance(parsed, dict) and isinstance(
            parsed.get("completed_goal_results"), list
        ):
            return _serialize_approved_result(request.approved_execution_result)
        return json.dumps(parsed, ensure_ascii=False, indent=2, allow_nan=False)


class ScriptedFinalOutputProvider:
    """Return fixed output strings and record isolated requests for offline tests."""

    def __init__(self, *, candidates: list[str], repairs: list[str]) -> None:
        self._candidates = list(candidates)
        self._repairs = list(repairs)
        self.generation_requests: list[FinalGenerationRequest] = []
        self.repair_requests: list[OutputRepairRequest] = []

    def generate(self, request: FinalGenerationRequest) -> str:
        self.generation_requests.append(request)
        if not self._candidates:
            raise RuntimeError("No scripted final-answer candidate remains")
        return self._candidates.pop(0)

    def repair(self, request: OutputRepairRequest) -> str:
        self.repair_requests.append(request)
        if not self._repairs:
            raise RuntimeError("No scripted output-repair response remains")
        return self._repairs.pop(0)


def build_scripted_output_provider(scenario: str) -> FinalOutputProvider:
    """Build output behavior for deterministic offline demo scenarios."""
    deterministic = DeterministicFinalOutputProvider()
    if scenario in {
        "happy",
        "replan",
        "max-replan",
        "valid-json",
        "trusted-tools-success",
        "generated-python-success",
        "generated-python-repair",
        "generated-python-failure",
    }:
        return deterministic

    repaired = json.dumps(
        {
            "status": "completed",
            "answer": "Mean is 13.",
            "key_results": {
                "mean": 13,
                "sample_standard_error": 1.291,
                "n_observations": 4,
            },
            "limitations": [],
        },
        indent=2,
    )
    if scenario == "output-repair":
        return ScriptedFinalOutputProvider(
            candidates=[
                json.dumps(
                    {
                        "status": "completed",
                        "answer": "Mean is 13.",
                        "key_results": {
                            "mean": 13,
                            "sample_standard_error": 1.291,
                            "n_observations": 4,
                        },
                    },
                    indent=2,
                )
            ],
            repairs=[repaired],
        )
    if scenario == "malformed-json":
        return ScriptedFinalOutputProvider(
            candidates=[
                'Final answer:\n{"status": "completed", "answer": "Mean is 13."}'
            ],
            repairs=[repaired],
        )
    if scenario == "output-failure":
        return ScriptedFinalOutputProvider(
            candidates=['{"status":"completed","answer":"Mean is 13."}'],
            repairs=["still not JSON"],
        )
    raise ValueError(f"Unknown offline output scenario: {scenario}")
