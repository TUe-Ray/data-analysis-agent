"""Bounded, formatting-only final-output generation and repair helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

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


def _serialize_approved_result(approved_execution_result: str) -> str:
    answer = FinalAnswer(
        status="completed",
        answer=approved_execution_result.strip(),
        key_results=_extract_key_results(approved_execution_result),
        limitations=[],
    )
    return json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2)


class DeterministicFinalOutputProvider:
    """Format approved content without asking a model to judge or recalculate it."""

    def generate(self, request: FinalGenerationRequest) -> str:
        return _serialize_approved_result(request.approved_execution_result)

    def repair(self, request: OutputRepairRequest) -> str:
        return _serialize_approved_result(request.approved_execution_result)


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
    if scenario in {"happy", "replan", "max-replan", "valid-json"}:
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
