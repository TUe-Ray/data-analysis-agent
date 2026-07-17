from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

TOLERANCE = 0.0011
REFERENCE_PATH = Path(__file__).with_name("reference.json")


def _compare(expected: Any, actual: Any, path: str, errors: list[str]) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if expected != actual:
            errors.append(f"{path}: expected {expected!r}, got {actual!r}")
        return
    if isinstance(expected, int):
        if type(actual) is not int or expected != actual:
            errors.append(f"{path}: expected integer {expected}, got {actual!r}")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            errors.append(f"{path}: expected number {expected}, got {actual!r}")
        elif not math.isfinite(float(actual)) or abs(float(actual) - expected) > TOLERANCE:
            errors.append(f"{path}: expected {expected}, got {actual}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            errors.append(f"{path}: expected list, got {type(actual).__name__}")
            return
        if path.endswith("selected_pairs"):
            key = lambda row: (row.get("study_id"), row.get("analysis_subject_id"))
            expected = sorted(expected, key=key)
            actual = sorted(actual, key=key)
        if len(expected) != len(actual):
            errors.append(f"{path}: expected length {len(expected)}, got {len(actual)}")
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{index}]", errors)
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"{path}: expected object, got {type(actual).__name__}")
            return
        if set(expected) != set(actual):
            errors.append(f"{path}: expected keys {sorted(expected)}, got {sorted(actual)}")
        for key in expected.keys() & actual.keys():
            _compare(expected[key], actual[key], f"{path}.{key}", errors)
        return
    errors.append(f"{path}: unsupported expected type {type(expected).__name__}")


def grade(candidate: dict[str, Any]) -> dict[str, Any]:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not isinstance(candidate, dict):
        return {"score": 0.0, "passed": False, "errors": ["candidate is not an object"]}
    if candidate.get("status") not in {"completed", "completed_with_limitations"}:
        errors.append("status must be completed or completed_with_limitations")
    if not isinstance(candidate.get("answer"), str) or not candidate.get("answer", "").strip():
        errors.append("answer must be a non-empty string")
    if not isinstance(candidate.get("limitations"), list):
        errors.append("limitations must be a list")
    _compare(reference["key_results"], candidate.get("key_results"), "$.key_results", errors)
    return {"score": 1.0 if not errors else 0.0, "passed": not errors, "errors": errors}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    print(json.dumps(grade(candidate), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
