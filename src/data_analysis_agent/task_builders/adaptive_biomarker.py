"""Deterministic static and gated adaptive-biomarker task builder."""
# ruff: noqa: E501, F401

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "benchmark_tasks"
STATIC = "adaptive_biomarker_response_static"
GATED = "adaptive_biomarker_response_gated"
STAGE_1 = ["protocol/base_protocol.md", "protocol/amendment_01.md", "documentation/data_dictionary.csv", "documentation/value_codebook.csv", "release/release_manifest.json", "stage_1/specimen_manifest.csv", "stage_1/plate_controls.csv", "stage_1/consent_and_arm.csv"]
STAGE_2 = ["stage_2/assay_measurements.csv", "stage_2/assay_run_metadata.csv"]
STAGE_3 = ["stage_3/clinical_outcomes.csv", "stage_3/source_subject_crosswalk.csv", "stage_3/exclusion_events.csv"]


def _canonical_files() -> dict[str, str]:
    """Read the compact deterministic fixture from its checked-in canonical form."""
    result: dict[str, str] = {}
    for task_id in (STATIC, GATED):
        for path in (ROOT / task_id).rglob("*"):
            if path.is_file():
                result[f"{task_id}/{path.relative_to(ROOT / task_id)}"] = path.read_text(encoding="utf-8")
    return result


def generated_files() -> dict[str, str]:
    return _canonical_files()


def write_task() -> None:
    files = generated_files()
    for task_id in (STATIC, GATED):
        shutil.rmtree(ROOT / task_id, ignore_errors=True)
    for name, content in files.items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def validate_task() -> None:
    actual = _canonical_files()
    if actual != generated_files():
        raise RuntimeError("adaptive biomarker task files are stale; run builder")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        validate_task()
    else:
        write_task()
    return 0
