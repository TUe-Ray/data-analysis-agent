"""Private deterministic grader for the successive-difference smoke task."""

from data_analysis_agent.benchmark_grading import numeric_match
from data_analysis_agent.benchmark_types import GradeResult


def grade(candidate: dict, reference: dict) -> GradeResult:
    errors: list[str] = []
    required = {"status", "answer", "key_results", "limitations"}
    missing = sorted(required - candidate.keys())
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
    key_results = candidate.get("key_results")
    if not isinstance(key_results, dict):
        errors.append("missing required field: key_results")
    elif "mean_absolute_successive_difference" not in key_results:
        errors.append("missing required field: mean_absolute_successive_difference")
    elif not numeric_match(
        key_results["mean_absolute_successive_difference"],
        reference["mean_absolute_successive_difference"],
        float(reference["absolute_tolerance"]),
    ):
        errors.append("numerical mismatch: mean_absolute_successive_difference")
    passed = not errors
    return GradeResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        errors=errors,
        details={"error_category": "none" if passed else "answer_mismatch"},
    )
