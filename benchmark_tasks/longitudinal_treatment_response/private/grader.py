"""Deterministic grader for the longitudinal treatment-response task."""

from __future__ import annotations

import math
from typing import Any

from data_analysis_agent.benchmark_types import GradeResult


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _compare(
    actual: Any, expected: Any, tolerance: float, path: str, errors: list[str]
) -> int:
    """Return matched leaf count and append concise mismatch paths."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"type mismatch: {path}")
            return 0
        extra_keys = sorted(set(actual) - set(expected))
        for key in extra_keys:
            child = f"{path}.{key}" if path else key
            errors.append(f"unexpected field: {child}")
        matched = 0
        for key, expected_value in expected.items():
            child = f"{path}.{key}" if path else key
            if key not in actual:
                errors.append(f"missing field: {child}")
                continue
            matched += _compare(actual[key], expected_value, tolerance, child, errors)
        return matched
    if isinstance(expected, list):
        if not isinstance(actual, list):
            errors.append(f"type mismatch: {path}")
            return 0
        matched = 0
        if len(actual) != len(expected):
            errors.append(f"length mismatch: {path}")
        for index, expected_value in enumerate(expected):
            child = f"{path}[{index}]"
            if index >= len(actual):
                errors.append(f"missing item: {child}")
                continue
            matched += _compare(actual[index], expected_value, tolerance, child, errors)
        return matched
    if _is_number(expected):
        if not _is_number(actual) or abs(float(actual) - float(expected)) > tolerance:
            errors.append(f"numerical mismatch: {path}")
            return 0
        return 1
    if actual != expected:
        errors.append(f"value mismatch: {path}")
        return 0
    return 1


def _leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_leaf_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_leaf_count(item) for item in value)
    return 1


def grade(candidate: dict, reference: dict) -> GradeResult:
    errors: list[str] = []
    required_top = {"status", "answer", "key_results", "limitations"}
    missing_top = sorted(required_top - set(candidate))
    if missing_top:
        errors.append("missing required top-level fields: " + ", ".join(missing_top))
    extra_top = sorted(set(candidate) - required_top)
    if extra_top:
        errors.append("unexpected top-level fields: " + ", ".join(extra_top))

    if candidate.get("status") not in {"completed", "completed_with_limitations"}:
        errors.append("invalid status")
    if not isinstance(candidate.get("answer"), str):
        errors.append("answer must be a string")
    if not isinstance(candidate.get("limitations"), list):
        errors.append("limitations must be a list")

    actual = candidate.get("key_results")
    expected = reference["key_results"]
    tolerance = float(reference["absolute_tolerance"])
    if not isinstance(actual, dict):
        errors.append("missing required field: key_results")
        matched = 0
    else:
        matched = _compare(actual, expected, tolerance, "key_results", errors)

    total = _leaf_count(expected)
    score = matched / total if total else 0.0
    # Strict benchmark pass: all expected fields and values must match.
    passed = not errors
    categories = []
    if any("attrition" in error for error in errors):
        categories.append("attrition")
    if any("arm_statistics" in error for error in errors):
        categories.append("arm_statistics")
    if any("between_arm_comparison" in error for error in errors):
        categories.append("between_arm_comparison")
    if any("selected_pairs" in error for error in errors):
        categories.append("selected_pairs")
    if not categories and errors:
        categories.append("schema")

    return GradeResult(
        passed=passed,
        score=score,
        errors=errors[:50],
        details={
            "error_category": "none" if passed else ",".join(categories),
            "matched_reference_leaves": matched,
            "total_reference_leaves": total,
        },
    )
