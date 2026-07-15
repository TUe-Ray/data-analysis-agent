"""Deterministic grading primitives and private task-grader loading."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from data_analysis_agent.benchmark_types import GradeResult, PrivateGradingSpec


class TaskGrader(Protocol):
    """Interface implemented by ordinary task-specific Python graders."""

    def __call__(
        self,
        candidate: dict[str, JsonValue],
        reference: dict[str, JsonValue],
    ) -> GradeResult | dict[str, JsonValue]: ...


def numeric_match(actual: object, expected: object, tolerance: float) -> bool:
    """Compare finite numeric values with a deterministic absolute tolerance."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return False
    if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
        return False
    return abs(float(actual) - float(expected)) <= tolerance


def compare_values(
    actual: object,
    expected: object,
    *,
    tolerance: float = 0.0,
    unordered: bool = False,
) -> bool:
    """Support exact, numeric-tolerance, ordered-list, and set-like comparisons."""
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return numeric_match(actual, expected, tolerance)
    if unordered and isinstance(actual, list) and isinstance(expected, list):
        return sorted(actual, key=repr) == sorted(expected, key=repr)
    return actual == expected


def invalid_candidate_grade(error: str) -> GradeResult:
    """Return the same deterministic failure shape for parse/runtime absence."""
    return GradeResult(
        passed=False,
        score=0.0,
        errors=[error],
        details={"error_category": "invalid_candidate"},
    )


def grade_candidate(
    candidate: dict[str, JsonValue] | None,
    private: PrivateGradingSpec,
    *,
    candidate_error: str | None = None,
) -> GradeResult:
    """Load private material only after candidate production and grade once."""
    if candidate is None:
        return invalid_candidate_grade(
            candidate_error or "candidate JSON is unavailable"
        )
    reference_path = Path(private.reference_path)
    grader_path = Path(private.grader_path)
    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        spec = importlib.util.spec_from_file_location(
            f"benchmark_grader_{grader_path.parent.parent.name}", grader_path
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not create grader module specification")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        grader: Callable[..., object] = module.grade
        raw = grader(candidate, reference)
        return raw if isinstance(raw, GradeResult) else GradeResult.model_validate(raw)
    except Exception as error:
        return GradeResult(
            passed=False,
            score=0.0,
            errors=[f"grader error: {type(error).__name__}: {error}"],
            details={"error_category": "grader_error"},
        )
