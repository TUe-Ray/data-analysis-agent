"""Build and validate the distributed longitudinal benchmark task deterministically."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from data_analysis_agent.benchmark_types import GradeResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "benchmark_tasks/longitudinal_treatment_response"
TARGET_ROOT = (
    PROJECT_ROOT / "benchmark_tasks/longitudinal_treatment_response_distributed"
)

PUBLIC_FILES = [
    "protocol/study_protocol.md",
    "protocol/protocol_amendment_01.md",
    "documentation/data_dictionary.csv",
    "documentation/value_codebook.csv",
    "data/subjects.csv",
    "data/visit_catalog.csv",
    "data/measurements.csv",
    "data/exclusion_events.csv",
    "data/subject_crosswalk.csv",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_text(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _normalized(value: str) -> str:
    return value.strip().casefold().replace("_", " ")


def _arm_code(value: str) -> str:
    normalized = _normalized(value)
    if normalized in {"a", "arm a", "treatment a"}:
        return "ARM_A"
    if normalized in {"b", "arm b", "treatment b"}:
        return "ARM_B"
    return "ARM_OTHER" if normalized else ""


def _consent_code(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized == "yes":
        return "CONSENT_Y"
    if normalized == "no":
        return "CONSENT_N"
    return ""


def _status_code(value: str) -> str:
    return {
        "valid": "STATUS_VALID",
        "reviewed": "STATUS_REVIEWED",
        "invalid_sensor": "STATUS_SENSOR_FAIL",
        "rejected": "STATUS_REJECTED",
    }[value.strip().casefold()]


def _source_code(value: str) -> str:
    return {"edc": "SOURCE_EDC", "manual": "SOURCE_MANUAL"}[value.strip().casefold()]


def _site_code(value: str) -> str:
    return f"SITE_{value.strip().upper()}"


def _event_code(value: str) -> str:
    return {
        "pre-existing contraindication": "EXCL_CONTRAINDICATION",
        "protocol discontinuation": "EXCL_DISCONTINUATION",
        "later follow-up note": "EXCL_FOLLOWUP_NOTE",
        "late administrative exclusion": "EXCL_ADMINISTRATIVE",
    }[value.strip().casefold()]


def _source_subject(patient_id: str) -> str:
    return f"SRC-{patient_id}"


def generated_files() -> dict[str, str]:
    """Return every generated file from the original task's public inputs."""
    source_data = SOURCE_ROOT / "public/data"
    patients = _read_csv(source_data / "patients.csv")
    visits = _read_csv(source_data / "visits.csv")
    exclusions = _read_csv(source_data / "exclusions.csv")

    subjects = [
        {
            "subject_key": _source_subject(row["patient_id"]),
            "age_years": row["age"],
            "consent_code": _consent_code(row["consent"]),
            "assigned_group_code": _arm_code(row["treatment_arm"]),
            "index_date": row["treatment_start_date"],
            "recruiting_site_code": _site_code(row["site"]),
        }
        for row in patients
    ]
    crosswalk = [
        {
            "source_subject_key": _source_subject(row["patient_id"]),
            "analysis_subject_id": row["patient_id"],
        }
        for row in sorted(patients, key=lambda item: item["patient_id"])
    ]
    catalog: list[dict[str, str]] = []
    measurements: list[dict[str, str]] = []
    for index, row in enumerate(visits, start=1):
        encounter = f"ENC{index:04d}"
        catalog.append(
            {
                "encounter_key": encounter,
                "source_record_id": row["record_id"],
                "source_subject_key": _source_subject(row["patient_id"]),
                "observed_at": row["measurement_date"],
                "measurement_status_code": _status_code(row["measurement_status"]),
                "origin_system_code": _source_code(row["source_system"]),
            }
        )
        measurements.append(
            {
                "encounter_key": encounter,
                "response_value": row["value"],
                "qc_score": row["quality_score"],
            }
        )
    events = [
        {
            "event_key": row["exclusion_id"],
            "source_subject_key": _source_subject(row["patient_id"]),
            "event_effective_date": row["effective_date"],
            "event_code": _event_code(row["reason"]),
        }
        for row in exclusions
    ]

    source_config = json.loads(
        (SOURCE_ROOT / "public/task.json").read_text(encoding="utf-8")
    )
    task_config = {
        "public_files": PUBLIC_FILES,
        "answer_schema": source_config["answer_schema"],
        "metadata": {
            "description": (
                "Distributed-specification longitudinal cohort reconstruction"
            ),
            "domain": "clinical data analysis",
            "difficulty": "distributed multi-step workflow",
            "source_task_id": "longitudinal_treatment_response",
            "document_files": PUBLIC_FILES[:4],
            "document_precedence": [
                "protocol/protocol_amendment_01.md overrides conflicting sections "
                "of protocol/study_protocol.md"
            ],
        },
    }
    return {
        "public/task.json": json.dumps(task_config, indent=2) + "\n",
        "public/data/subjects.csv": _csv_text(
            [
                "subject_key",
                "age_years",
                "consent_code",
                "assigned_group_code",
                "index_date",
                "recruiting_site_code",
            ],
            subjects,
        ),
        "public/data/visit_catalog.csv": _csv_text(
            [
                "encounter_key",
                "source_record_id",
                "source_subject_key",
                "observed_at",
                "measurement_status_code",
                "origin_system_code",
            ],
            catalog,
        ),
        "public/data/measurements.csv": _csv_text(
            ["encounter_key", "response_value", "qc_score"], measurements
        ),
        "public/data/exclusion_events.csv": _csv_text(
            [
                "event_key",
                "source_subject_key",
                "event_effective_date",
                "event_code",
            ],
            events,
        ),
        "public/data/subject_crosswalk.csv": _csv_text(
            ["source_subject_key", "analysis_subject_id"], crosswalk
        ),
        "private/reference.json": (SOURCE_ROOT / "private/reference.json").read_text(
            encoding="utf-8"
        ),
        "private/grader.py": (SOURCE_ROOT / "private/grader.py").read_text(
            encoding="utf-8"
        ),
    }


def write_task() -> None:
    """Write deterministic generated tables/config and copied grading assets."""
    for relative_name, content in generated_files().items():
        path = TARGET_ROOT / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _finite(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def compute_oracle() -> dict[str, Any]:
    """Compute the distributed task answer solely from its public files."""
    data_root = TARGET_ROOT / "public/data"
    subjects = _read_csv(data_root / "subjects.csv")
    catalog = _read_csv(data_root / "visit_catalog.csv")
    measurements = _read_csv(data_root / "measurements.csv")
    events = _read_csv(data_root / "exclusion_events.csv")
    crosswalk_rows = _read_csv(data_root / "subject_crosswalk.csv")
    crosswalk = {
        row["source_subject_key"]: row["analysis_subject_id"] for row in crosswalk_rows
    }
    measurement_by_encounter = {row["encounter_key"]: row for row in measurements}

    joined: list[dict[str, str]] = []
    for visit in catalog:
        measurement = measurement_by_encounter[visit["encounter_key"]]
        joined.append(
            {
                "record_id": visit["source_record_id"],
                "patient_id": crosswalk[visit["source_subject_key"]],
                "measurement_date": visit["observed_at"],
                "value": measurement["response_value"],
                "status": visit["measurement_status_code"],
                "quality_score": measurement["qc_score"],
                "source_system": visit["origin_system_code"],
            }
        )
    logical_columns = [
        "record_id",
        "patient_id",
        "measurement_date",
        "value",
        "status",
        "quality_score",
        "source_system",
    ]
    seen: set[tuple[str, ...]] = set()
    deduplicated: list[dict[str, str]] = []
    for row in joined:
        key = tuple(row[column] for column in logical_columns)
        if key not in seen:
            seen.add(key)
            deduplicated.append(row)

    valid_visits: list[dict[str, Any]] = []
    for row in deduplicated:
        observed = _date(row["measurement_date"])
        value = _finite(row["value"])
        quality = _finite(row["quality_score"])
        if (
            row["status"] in {"STATUS_VALID", "STATUS_REVIEWED"}
            and observed is not None
            and value is not None
            and quality is not None
        ):
            valid_visits.append(
                {**row, "date": observed, "numeric_value": value, "quality": quality}
            )

    subjects_by_id: dict[str, dict[str, Any]] = {}
    for row in subjects:
        patient_id = crosswalk[row["subject_key"]]
        start = _date(row["index_date"])
        age = _finite(row["age_years"])
        arm = {"ARM_A": "A", "ARM_B": "B"}.get(row["assigned_group_code"])
        if (
            age is not None
            and 18 <= age <= 75
            and row["consent_code"] == "CONSENT_Y"
            and arm is not None
            and start is not None
        ):
            subjects_by_id[patient_id] = {**row, "start": start, "arm": arm}

    events_by_id: dict[str, list[date]] = defaultdict(list)
    for row in events:
        effective = _date(row["event_effective_date"])
        if effective is not None:
            events_by_id[crosswalk[row["source_subject_key"]]].append(effective)
    visits_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_visits:
        visits_by_id[row["patient_id"]].append(row)

    pre_start_removed: set[str] = set()
    baseline_missing: set[str] = set()
    baselines: dict[str, dict[str, Any]] = {}
    for patient_id, subject in subjects_by_id.items():
        if any(day <= subject["start"] for day in events_by_id[patient_id]):
            pre_start_removed.add(patient_id)
            continue
        candidates = []
        for visit in visits_by_id[patient_id]:
            relative_day = (visit["date"] - subject["start"]).days
            if -14 <= relative_day <= -1:
                candidates.append((relative_day, visit["quality"], visit))
        if not candidates:
            baseline_missing.add(patient_id)
            continue
        baselines[patient_id] = sorted(
            candidates,
            key=lambda item: (-item[0], -item[1], item[2]["record_id"]),
        )[0][2]

    followup_missing: set[str] = set()
    followups: dict[str, dict[str, Any]] = {}
    for patient_id in baselines:
        subject = subjects_by_id[patient_id]
        candidates = []
        for visit in visits_by_id[patient_id]:
            relative_day = (visit["date"] - subject["start"]).days
            if 28 <= relative_day <= 42:
                candidates.append((relative_day, visit))
        if not candidates:
            followup_missing.add(patient_id)
            continue
        followups[patient_id] = sorted(
            candidates,
            key=lambda item: (
                abs(item[0] - 35),
                -item[1]["quality"],
                item[1]["date"],
                item[1]["record_id"],
            ),
        )[0][1]

    post_start_removed: set[str] = set()
    selected_pairs: list[dict[str, Any]] = []
    for patient_id, followup in followups.items():
        subject = subjects_by_id[patient_id]
        if any(
            subject["start"] < day <= followup["date"]
            for day in events_by_id[patient_id]
        ):
            post_start_removed.add(patient_id)
            continue
        baseline = baselines[patient_id]
        selected_pairs.append(
            {
                "patient_id": patient_id,
                "arm": subject["arm"],
                "baseline_record_id": baseline["record_id"],
                "followup_record_id": followup["record_id"],
                "baseline_value": round(baseline["numeric_value"], 3),
                "followup_value": round(followup["numeric_value"], 3),
                "change": round(
                    followup["numeric_value"] - baseline["numeric_value"], 3
                ),
            }
        )
    selected_pairs.sort(key=lambda row: row["patient_id"])

    arm_statistics: dict[str, dict[str, float | int]] = {}
    raw_mean_changes: dict[str, float] = {}
    for arm in ("A", "B"):
        arm_rows = [row for row in selected_pairs if row["arm"] == arm]
        changes = [float(row["change"]) for row in arm_rows]
        sample_sd = statistics.stdev(changes)
        raw_mean_changes[arm] = statistics.mean(changes)
        arm_statistics[arm] = {
            "n": len(arm_rows),
            "mean_baseline": round(
                statistics.mean(float(row["baseline_value"]) for row in arm_rows), 3
            ),
            "mean_followup": round(
                statistics.mean(float(row["followup_value"]) for row in arm_rows), 3
            ),
            "mean_change": round(raw_mean_changes[arm], 3),
            "sample_sd_change": round(sample_sd, 3),
            "sample_se_change": round(sample_sd / math.sqrt(len(arm_rows)), 3),
        }
    difference = raw_mean_changes["B"] - raw_mean_changes["A"]
    return {
        "status": "completed",
        "answer": (
            "Descriptive treatment response computed from reconciled public data."
        ),
        "key_results": {
            "attrition": {
                "total_patients": len(subjects),
                "basic_ineligible": len(subjects) - len(subjects_by_id),
                "eligible_after_basic_checks": len(subjects_by_id),
                "excluded_pre_start": len(pre_start_removed),
                "no_valid_baseline": len(baseline_missing),
                "no_valid_followup": len(followup_missing),
                "excluded_post_start_before_or_on_followup": len(post_start_removed),
                "complete_pairs": len(selected_pairs),
                "complete_pairs_arm_a": sum(
                    row["arm"] == "A" for row in selected_pairs
                ),
                "complete_pairs_arm_b": sum(
                    row["arm"] == "B" for row in selected_pairs
                ),
                "exact_duplicate_visit_rows_removed": len(joined) - len(deduplicated),
                "invalid_or_missing_visit_rows_excluded": len(deduplicated)
                - len(valid_visits),
            },
            "arm_statistics": arm_statistics,
            "between_arm_comparison": {
                "difference_in_mean_change_b_minus_a": round(difference, 3)
            },
            "selected_pairs": selected_pairs,
        },
        "limitations": [
            "Descriptive analysis only; no causal claim or hypothesis test."
        ],
    }


def _grade_oracle(candidate: dict[str, Any]) -> GradeResult:
    grader_path = TARGET_ROOT / "private/grader.py"
    reference = json.loads(
        (TARGET_ROOT / "private/reference.json").read_text(encoding="utf-8")
    )
    spec = importlib.util.spec_from_file_location(
        "distributed_task_grader", grader_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load distributed grader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grade(candidate, reference)


def validate_task() -> None:
    """Fail if generated assets drift or public analysis differs from reference."""
    errors = []
    for relative_name, expected in generated_files().items():
        path = TARGET_ROOT / relative_name
        actual = path.read_text(encoding="utf-8") if path.is_file() else None
        if actual != expected:
            errors.append(relative_name)
    if errors:
        raise RuntimeError("generated task files are stale: " + ", ".join(errors))
    candidate = compute_oracle()
    grade = _grade_oracle(candidate)
    if not grade.passed:
        raise RuntimeError(
            "distributed oracle failed private grading: " + repr(grade.errors)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate task assets")
    mode.add_argument("--check", action="store_true", help="validate committed assets")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.write:
        write_task()
    validate_task()
    print("Distributed longitudinal task is deterministic and oracle-valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
