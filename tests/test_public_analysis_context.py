"""Fair public-context coverage for every code-based benchmark architecture."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from data_analysis_agent.benchmark import DEFAULT_TASKS_ROOT
from data_analysis_agent.benchmark_approaches import (
    _resolved_public_files,
    build_one_shot_code_messages,
    build_single_agent_messages,
)
from data_analysis_agent.benchmark_context import build_public_analysis_context
from data_analysis_agent.benchmark_tasks import load_benchmark_task, stage_public_task
from data_analysis_agent.benchmark_types import PublicTaskView
from data_analysis_agent.prompts import (
    build_planner_messages,
    build_python_repair_messages,
)

TASK_ID = "longitudinal_treatment_response_distributed"


def _distributed_context(tmp_path: Path):
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, TASK_ID)
    attempt = tmp_path / "attempt"
    public = stage_public_task(task.public, attempt)
    staged = _resolved_public_files(public, attempt)
    return task, public, staged, build_public_analysis_context(public, staged)


def test_shared_context_contains_documents_profiles_paths_and_no_private_data(
    tmp_path: Path,
) -> None:
    task, public, staged, context = _distributed_context(tmp_path)
    documents = {
        item["public_relative_filename"]: item
        for item in context["specification_documents"]
    }
    declared = public.metadata["document_files"]

    assert set(documents) == set(declared)
    for name in declared:
        assert documents[name]["content"] == public.data_contents[f"inputs/{name}"]
    profiles = {
        item["public_relative_filename"]: item for item in context["csv_profiles"]
    }
    expected_csvs = {
        name for name in public.metadata["document_files"] if name.endswith(".csv")
    } | {
        name.removeprefix("inputs/")
        for name in public.data_files
        if name.endswith(".csv")
    }
    assert set(profiles) == expected_csvs
    assert {item["staged_path"] for item in profiles.values()} <= {
        str(path.resolve()) for path in staged
    }
    assert all(len(item["representative_rows"]) <= 3 for item in profiles.values())

    measurements = profiles["data/measurements.csv"]
    rows = list(
        csv.DictReader(
            io.StringIO(public.data_contents["inputs/data/measurements.csv"])
        )
    )
    assert measurements["row_count"] == len(rows)
    assert rows[3] not in measurements["representative_rows"]
    serialized = json.dumps(context, ensure_ascii=False)
    assert rows[3]["encounter_key"] not in serialized
    assert (
        Path(task.private.reference_path).read_text(encoding="utf-8") not in serialized
    )
    assert "private/reference.json" not in serialized
    assert "private/grader.py" not in serialized
    assert context["task"]["document_precedence"]


def test_first_code_generation_and_planning_prompts_share_identical_facts(
    tmp_path: Path,
) -> None:
    _, public, staged, context = _distributed_context(tmp_path)
    serialized = json.dumps(context, ensure_ascii=False)
    one_shot = build_one_shot_code_messages(public, context)[1]["content"]
    single = build_single_agent_messages(public, tmp_path / "single", staged, context)[
        1
    ]["content"]
    planner = build_planner_messages(
        question=public.prompt,
        input_context=serialized,
        replan_count=0,
        answer_schema=public.answer_schema,
        max_plan_goals=6,
    )[1]["content"]

    for prompt in (one_shot, single, planner):
        assert serialized in prompt
        assert "Where this amendment conflicts" in prompt
        assert all(str(path.resolve()) in prompt for path in staged)
    # single_agent_checker uses the same single-agent core and therefore this exact
    # initial generation message; checker behavior is covered separately.


def test_initial_generation_and_mechanical_repair_share_specification_facts(
    tmp_path: Path,
) -> None:
    _, public, staged, context = _distributed_context(tmp_path)
    serialized = json.dumps(context, ensure_ascii=False)
    initial = build_single_agent_messages(public, tmp_path / "single", staged, context)[
        1
    ]["content"]
    repair = build_python_repair_messages(
        current_goal={
            "goal_id": "analysis",
            "objective": "Analyze the public task.",
            "required_outputs": ["answer"],
            "constraints": [],
            "success_criteria": ["complete"],
            "depends_on": [],
        },
        code="raise RuntimeError('fixture')\n",
        failure_category="runtime_error",
        stdout="",
        stderr="fixture",
        error="RuntimeError: fixture",
        staged_file_paths=[str(path) for path in staged],
        goal_directory=str(tmp_path / "single"),
        input_context=serialized,
    )[1]["content"]

    for fact in (
        "Where this amendment conflicts",
        "STATUS_REVIEWED",
        "analysis_subject_id",
    ):
        assert fact in initial
        assert fact in repair


def test_legacy_task_without_document_metadata_builds_a_compact_context(
    tmp_path: Path,
) -> None:
    staged = (tmp_path / "inputs/data.csv").resolve()
    staged.parent.mkdir(parents=True)
    staged.write_text("value\n1\n2\n", encoding="utf-8")
    public = PublicTaskView(
        task_id="legacy",
        prompt="Sum value.",
        data_files=["inputs/data.csv"],
        data_contents={"inputs/data.csv": "value\n1\n2\n"},
        answer_schema={"type": "object"},
    )

    context = build_public_analysis_context(public, [staged])

    assert context["specification_documents"] == []
    assert context["csv_profiles"][0]["row_count"] == 2
    assert context["csv_profiles"][0]["staged_path"] == str(staged)
