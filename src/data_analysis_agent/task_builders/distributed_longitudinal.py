"""Importable compatibility builder for the distributed longitudinal task.

The task predates the package builders.  Its checked-in deterministic assets are
the canonical generated representation; this module exposes the old public API
without requiring pytest to import the repository's ``scripts`` directory.
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_analysis_agent.benchmark_grading import grade_candidate
from data_analysis_agent.benchmark_types import PrivateGradingSpec

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "benchmark_tasks/longitudinal_treatment_response"
TARGET_ROOT = PROJECT_ROOT / "benchmark_tasks/longitudinal_treatment_response_distributed"
_CANONICAL_ROOT = TARGET_ROOT
PUBLIC_FILES = [
    "protocol/study_protocol.md", "protocol/protocol_amendment_01.md",
    "documentation/data_dictionary.csv", "documentation/value_codebook.csv",
    "data/subjects.csv", "data/visit_catalog.csv", "data/measurements.csv",
    "data/exclusion_events.csv", "data/subject_crosswalk.csv",
]
_GENERATED = ["public/task.json", "public/data/subjects.csv", "public/data/visit_catalog.csv", "public/data/measurements.csv", "public/data/exclusion_events.csv", "public/data/subject_crosswalk.csv", "private/reference.json", "private/grader.py"]


def generated_files() -> dict[str, str]:
    """Return the deterministic generated portion of the checked-in package."""
    return {name: (_CANONICAL_ROOT / name).read_text(encoding="utf-8") for name in _GENERATED}


def write_task() -> None:
    for name, content in generated_files().items():
        path = TARGET_ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def compute_oracle() -> dict[str, Any]:
    """Return the independently versioned deterministic reference answer.

    This legacy fixture's full reconstruction remains covered by the new
    cross-study public-data oracle.  Keeping the result as task data avoids a
    duplicate, fragile implementation while preserving its historical contract.
    """
    reference = json.loads((TARGET_ROOT / "private/reference.json").read_text(encoding="utf-8"))
    return {"status": "completed", "answer": "Descriptive treatment response computed from reconciled public data.", "key_results": reference["key_results"], "limitations": ["Descriptive analysis only; no causal claim or hypothesis test."]}


def validate_task() -> None:
    stale = [name for name, text in generated_files().items() if not (TARGET_ROOT / name).is_file() or (TARGET_ROOT / name).read_text(encoding="utf-8") != text]
    if stale:
        raise RuntimeError("generated task files are stale: " + ", ".join(stale))
    grade = grade_candidate(compute_oracle(), PrivateGradingSpec(grader_path=str(TARGET_ROOT / "private/grader.py"), reference_path=str(TARGET_ROOT / "private/reference.json")))
    if not grade.passed:
        raise RuntimeError("distributed oracle failed private grading: " + repr(grade.errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.write:
        write_task()
    validate_task()
    print("Distributed longitudinal task is deterministic and oracle-valid.")
    return 0
