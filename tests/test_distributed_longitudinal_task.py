"""Integration coverage for the distributed-specification interview task."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from data_analysis_agent.benchmark import DEFAULT_TASKS_ROOT
from data_analysis_agent.benchmark_approaches import (
    _agent_input_profile,
    _resolved_public_files,
    build_direct_answer_messages,
    build_single_agent_messages,
)
from data_analysis_agent.benchmark_grading import grade_candidate
from data_analysis_agent.benchmark_tasks import (
    BenchmarkTaskError,
    load_benchmark_task,
    stage_public_task,
)
from data_analysis_agent.benchmark_types import PrivateGradingSpec
from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import ScriptedRoleModel
from data_analysis_agent.nodes import _artifacts_available_to_current_goal
from data_analysis_agent.python_runner import LocalPythonRunner
from data_analysis_agent.task_builders import distributed_longitudinal as builder

TASK_ID = "longitudinal_treatment_response_distributed"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _task_root() -> Path:
    return DEFAULT_TASKS_ROOT / TASK_ID


def test_distributed_task_loads_and_stages_every_manifested_public_file(
    tmp_path: Path,
) -> None:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, TASK_ID)

    assert task.public.data_files == builder.PUBLIC_FILES
    staged = stage_public_task(task.public, tmp_path / "attempt")
    assert staged.data_files == [f"inputs/{name}" for name in builder.PUBLIC_FILES]
    assert all((tmp_path / "attempt" / path).is_file() for path in staged.data_files)


def test_private_material_never_enters_messages_profiles_or_allowlist(
    tmp_path: Path,
) -> None:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, TASK_ID)
    attempt = tmp_path / "attempt"
    public = stage_public_task(task.public, attempt)
    staged_paths = _resolved_public_files(public, attempt)
    direct = "\n".join(
        message["content"] for message in build_direct_answer_messages(public)
    )
    single = "\n".join(
        message["content"]
        for message in build_single_agent_messages(
            public, attempt / "single_agent_run", staged_paths
        )
    )
    profile = json.dumps(_agent_input_profile(public))
    private_reference = Path(task.private.reference_path).read_text(encoding="utf-8")

    assert len(staged_paths) == len(builder.PUBLIC_FILES)
    assert all((attempt / "inputs").resolve() in path.parents for path in staged_paths)
    for exposed in (direct, single, profile):
        assert "private/reference.json" not in exposed
        assert "private/grader.py" not in exposed
        assert private_reference not in exposed
    assert "Where this amendment conflicts" in profile
    assert "STATUS_REVIEWED" in profile


def test_public_manifest_rejects_private_path_traversal(tmp_path: Path) -> None:
    task_root = tmp_path / "unsafe"
    public = task_root / "public"
    private = task_root / "private"
    public.mkdir(parents=True)
    private.mkdir()
    (public / "prompt.txt").write_text("Unsafe fixture", encoding="utf-8")
    (public / "task.json").write_text(
        json.dumps(
            {
                "public_files": ["../private/reference.json"],
                "answer_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    (private / "reference.json").write_text("{}", encoding="utf-8")
    (private / "grader.py").write_text(
        "def grade(candidate, reference): ...\n", encoding="utf-8"
    )

    with pytest.raises(BenchmarkTaskError, match="Unsafe or missing public file"):
        load_benchmark_task(tmp_path, "unsafe")


def test_transformation_is_deterministic_and_does_not_modify_source_task(
    tmp_path: Path, monkeypatch
) -> None:
    source_files = sorted(
        path
        for path in builder.SOURCE_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    before = {
        path.relative_to(builder.SOURCE_ROOT): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source_files
    }
    first = builder.generated_files()
    second = builder.generated_files()
    assert first == second

    monkeypatch.setattr(builder, "TARGET_ROOT", tmp_path / TASK_ID)
    builder.write_task()
    after = {
        path.relative_to(builder.SOURCE_ROOT): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source_files
    }
    assert after == before
    assert all(
        (tmp_path / TASK_ID / relative_name).read_text(encoding="utf-8") == content
        for relative_name, content in first.items()
    )


def test_crosswalk_and_heterogeneous_join_cardinalities_are_valid() -> None:
    data = _task_root() / "public/data"
    crosswalk = _rows(data / "subject_crosswalk.csv")
    visits = _rows(data / "visit_catalog.csv")
    measurements = _rows(data / "measurements.csv")

    source_keys = [row["source_subject_key"] for row in crosswalk]
    analysis_ids = [row["analysis_subject_id"] for row in crosswalk]
    visit_encounters = [row["encounter_key"] for row in visits]
    measurement_encounters = [row["encounter_key"] for row in measurements]
    assert len(source_keys) == len(set(source_keys)) == 48
    assert len(analysis_ids) == len(set(analysis_ids)) == 48
    assert len(visit_encounters) == len(set(visit_encounters)) == 331
    assert len(measurement_encounters) == len(set(measurement_encounters)) == 331
    assert set(visit_encounters) == set(measurement_encounters)
    assert {row["source_subject_key"] for row in visits} <= set(source_keys)


def test_amendment_and_rule_equivalence_checklist_are_complete() -> None:
    public = _task_root() / "public"
    base = (public / "protocol/study_protocol.md").read_text(encoding="utf-8")
    amendment = (public / "protocol/protocol_amendment_01.md").read_text(
        encoding="utf-8"
    )
    amendment_words = " ".join(amendment.casefold().split())
    effective = " ".join((base + "\n" + amendment).casefold().split())

    assert "where this amendment conflicts" in amendment_words
    assert "all unaffected protocol sections remain in force" in amendment_words
    for override in (
        "day 28 through day 42 inclusive",
        "day 35",
        "valid` and `reviewed",
        "index_date < event_effective_date <= selected_followup_date",
    ):
        assert override in amendment
    checklist = {
        "age 18-75": ("18 through 75",),
        "baseline window": ("day -14", "day -1"),
        "follow-up window": ("day 28", "day 42"),
        "target": ("day 35",),
        "accepted statuses": ("valid", "reviewed"),
        "sample SD": ("n - 1",),
        "sample SE": ("sqrt(n)",),
        "comparison": ("arm b", "minus arm a"),
        "attrition": ("total_patients", "invalid_or_missing_visit_rows_excluded"),
        "selected pairs": ("baseline_record_id", "followup_record_id"),
        "rounding": ("three", "decimal"),
    }
    for name, terms in checklist.items():
        for term in terms:
            assert term in effective, f"effective rules omit {name}: {term}"


def test_reference_is_identical_and_public_oracle_passes_private_grader() -> None:
    original = load_benchmark_task(
        DEFAULT_TASKS_ROOT, "longitudinal_treatment_response"
    )
    distributed = load_benchmark_task(DEFAULT_TASKS_ROOT, TASK_ID)
    assert (
        Path(original.private.reference_path).read_bytes()
        == Path(distributed.private.reference_path).read_bytes()
    )
    assert (
        Path(original.private.grader_path).read_bytes()
        == Path(distributed.private.grader_path).read_bytes()
    )
    assert original.public.answer_schema == distributed.public.answer_schema

    candidate = builder.compute_oracle()
    grade = grade_candidate(
        candidate,
        PrivateGradingSpec(
            grader_path=distributed.private.grader_path,
            reference_path=distributed.private.reference_path,
        ),
    )
    assert grade.passed
    assert grade.score == 1.0


def test_distributed_artifact_is_visible_only_after_dependency_approval(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "goals/reconcile/normalized_subject_cohort.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("patient_id,arm\nP001,A\n", encoding="utf-8")
    plan = {
        "scientific_objective": "Analyze the distributed longitudinal task.",
        "goals": [
            {
                "goal_id": "reconcile",
                "objective": "Reconcile documents and normalize subjects.",
                "required_outputs": ["normalized cohort artifact"],
                "constraints": [],
                "success_criteria": ["cohort is complete"],
                "depends_on": [],
            },
            {
                "goal_id": "select_visits",
                "objective": "Select valid baseline and follow-up records.",
                "required_outputs": ["selected records"],
                "constraints": [],
                "success_criteria": ["records follow effective protocol"],
                "depends_on": ["reconcile"],
            },
        ],
        "final_output_goal_id": "select_visits",
    }
    artifact_manifest = {
        "artifact_id": "reconcile:normalized_subject_cohort.csv:0123456789ab",
        "producer_goal_id": "reconcile",
        "path": str(artifact),
        "relative_name": artifact.name,
        "media_type": "text/csv",
        "description": "Normalized eligible subject cohort.",
        "size_bytes": artifact.stat().st_size,
        "sha256": "0" * 64,
        "columns": ["patient_id", "arm"],
        "row_count": 1,
    }
    base_state = {
        "high_level_plan": plan,
        "current_goal": plan["goals"][1],
        "completed_goal_results": [{"goal_id": "reconcile"}],
    }

    assert (
        _artifacts_available_to_current_goal(
            {**base_state, "pending_goal_artifacts": [artifact_manifest]}
        )
        == []
    )
    available = _artifacts_available_to_current_goal(
        {**base_state, "approved_goal_artifacts": [artifact_manifest]}
    )
    assert [item.path for item in available] == [str(artifact)]


def test_full_graph_consumes_a_distributed_task_artifact_through_dependency(
    tmp_path: Path,
) -> None:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, TASK_ID)
    attempt = tmp_path / "attempt"
    public = stage_public_task(task.public, attempt)
    paths = {Path(path).name: (attempt / path).resolve() for path in public.data_files}
    run_directory = attempt / "agent_run"
    artifact = run_directory / "goals/reconcile/normalized_subject_cohort.csv"
    plan = json.dumps(
        {
            "scientific_objective": "Normalize subjects then consume the cohort.",
            "goals": [
                {
                    "goal_id": "reconcile",
                    "objective": "Resolve canonical subject identifiers.",
                    "required_outputs": ["normalized cohort artifact"],
                    "constraints": ["Use the public crosswalk."],
                    "success_criteria": ["Every subject resolves exactly once."],
                    "depends_on": [],
                },
                {
                    "goal_id": "consume",
                    "objective": "Consume the verified normalized cohort.",
                    "required_outputs": ["normalized subject count"],
                    "constraints": ["Use only a dependency-approved artifact."],
                    "success_criteria": ["All 48 normalized subjects are read."],
                    "depends_on": ["reconcile"],
                },
            ],
            "final_output_goal_id": "consume",
        }
    )
    strategy = json.dumps(
        {
            "strategy": "generated_python",
            "capability_name": None,
            "arguments": {},
            "concise_reason": "The goal requires a documented local join.",
        }
    )
    reconcile_source = "\n".join(
        [
            "import pandas as pd",
            f"subjects = pd.read_csv({str(paths['subjects.csv'])!r})",
            f"crosswalk = pd.read_csv({str(paths['subject_crosswalk.csv'])!r})",
            "normalized = subjects.merge(crosswalk, left_on='subject_key', "
            "right_on='source_subject_key', validate='one_to_one')",
            "normalized[['analysis_subject_id', 'assigned_group_code']].to_csv("
            "'normalized_subject_cohort.csv', index=False)",
            "__agent_result__ = {'normalized_rows': int(len(normalized)), "
            "'artifacts': [{'relative_name': 'normalized_subject_cohort.csv', "
            "'description': 'Canonical subject and group cohort', "
            "'media_type': 'text/csv'}]}",
        ]
    )
    consume_source = "\n".join(
        [
            "import pandas as pd",
            f"cohort = pd.read_csv({str(artifact)!r})",
            "__agent_result__ = {'normalized_rows_consumed': int(len(cohort))}",
        ]
    )

    def generation(source: str) -> str:
        return json.dumps(
            {"kind": "python", "code_lines": source.splitlines(), "summary": "Run."}
        )

    model = ScriptedRoleModel(
        {
            "planner": [plan],
            "executor": [
                strategy,
                generation(reconcile_source),
                strategy,
                generation(consume_source),
            ],
            "verifier": [
                '{"decision":"PASS","feedback":"Cohort artifact is verified."}',
                '{"decision":"PASS","feedback":"Approved cohort was consumed."}',
            ],
        }
    )
    result = build_graph(model, runner=LocalPythonRunner(timeout_seconds=10)).invoke(
        {
            "question": public.prompt,
            "file_paths": [Path(path).name for path in public.data_files],
            "staged_file_paths": [str(path) for path in paths.values()],
            "input_context": json.dumps(_agent_input_profile(public)),
            "run_directory": str(run_directory),
            "replan_count": 0,
            "max_replans": 1,
            "max_code_repair_attempts": 2,
            "max_code_repair_no_progress_attempts": 3,
            "trace": [],
        }
    )

    assert result["status"] == "completed"
    assert result["completed_goal_results"][-1]["result"] == {
        "normalized_rows_consumed": 48
    }
    assert result["approved_goal_artifacts"][0]["path"] == str(artifact)
    consume_call = [call for call in model.calls if call.role == "executor"][-1]
    assert str(artifact) in consume_call.messages[1]["content"]
