"""Deterministic coverage for gated public evidence and model normalization."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

from data_analysis_agent.benchmark_context import build_public_analysis_context
from data_analysis_agent.benchmark_tasks import load_benchmark_task, stage_public_task
from data_analysis_agent.nodes import (
    _normalized_json_object,
    _release_deferred_public_inputs,
)
from data_analysis_agent.schemas import GoalArtifact, PythonGeneration

BUILDER = runpy.run_path("scripts/build_adaptive_biomarker_tasks.py")
TASKS_ROOT = Path(__file__).resolve().parents[1] / "benchmark_tasks"


def test_builder_is_deterministic_and_gated_stage_is_not_initially_readable(
    tmp_path: Path,
) -> None:
    assert BUILDER["generated_files"]() == BUILDER["generated_files"]()
    task = load_benchmark_task(TASKS_ROOT, BUILDER["GATED"])
    public = stage_public_task(task.public, tmp_path)
    assert len(public.data_files) == len(BUILDER["STAGE_1"])
    assert not (tmp_path / "inputs/stage_2/assay_measurements.csv").exists()
    context = json.dumps(
        build_public_analysis_context(
            public, [(tmp_path / item).resolve() for item in public.data_files]
        )
    )
    assert "R1,S01,12" not in context
    assert "stage_2/assay_measurements.csv" in context


def test_verified_qc_columns_release_exactly_stage_two(tmp_path: Path) -> None:
    task = load_benchmark_task(TASKS_ROOT, BUILDER["GATED"])
    public = stage_public_task(task.public, tmp_path)
    paths = [(tmp_path / item).resolve() for item in public.data_files]
    artifact = GoalArtifact(
        artifact_id="qc:approved",
        producer_goal_id="any_goal_name",
        path=str(paths[0]),
        relative_name="qc_approved_specimens.csv",
        description="verified QC",
        size_bytes=1,
        sha256="0" * 64,
        columns=[
            "analysis_subject_id",
            "specimen_id",
            "plate_id",
            "arm_code",
            "qc_status",
        ],
        row_count=1,
    )
    state = {
        "staged_file_paths": [str(path) for path in paths],
        "staged_file_display_paths": list(public.data_files),
        "public_data_contents": dict(public.data_contents),
        "public_metadata": dict(public.metadata),
        "public_task_id": public.task_id,
        "public_prompt_variant": public.prompt_variant,
        "question": public.prompt,
        "answer_schema": public.answer_schema,
        "deferred_public_files": dict(public.deferred_public_files),
        "release_stages": list(public.release_stages),
        "release_history": [],
        "current_goal_result": {"result": {}},
    }
    update = _release_deferred_public_inputs(state, [artifact])
    assert update["release_history"][0]["stage"] == "stage_2"
    assert (tmp_path / "inputs/stage_2/assay_measurements.csv").is_file()
    assert not (tmp_path / "inputs/stage_3/clinical_outcomes.csv").exists()


def test_fenced_legacy_code_is_representation_normalized_only() -> None:
    raw = '```json\n{"kind":"python","code":"x = 1","summary":"x"}\n```'
    parsed = PythonGeneration.model_validate_json(
        _normalized_json_object(raw, allow_code=True)
    )
    assert parsed.source() == "x = 1\n"
