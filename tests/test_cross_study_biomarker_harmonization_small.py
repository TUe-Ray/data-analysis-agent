"""Integration, fairness, and strict-grader coverage for the static task."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from data_analysis_agent.benchmark import DEFAULT_TASKS_ROOT
from data_analysis_agent.benchmark_approaches import (
    _resolved_public_files,
    build_one_shot_code_messages,
    build_single_agent_messages,
)
from data_analysis_agent.benchmark_context import build_public_analysis_context
from data_analysis_agent.benchmark_grading import grade_candidate
from data_analysis_agent.benchmark_tasks import load_benchmark_task, stage_public_task
from data_analysis_agent.benchmark_types import PrivateGradingSpec
from data_analysis_agent.nodes import _cross_study_consistency_error
from data_analysis_agent.prompts import build_planner_messages
from data_analysis_agent.task_builders import cross_study_biomarker_small as builder

TASK_ROOT = DEFAULT_TASKS_ROOT / builder.TASK_ID


def test_static_task_is_deterministic_oracle_valid_and_fully_staged(
    tmp_path: Path,
) -> None:
    builder.check_task(TASK_ROOT)
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, builder.TASK_ID)
    public = stage_public_task(task.public, tmp_path / "attempt")

    assert task.public.deferred_public_files == {}
    assert task.public.release_stages == []
    assert all(
        (tmp_path / "attempt" / staged_path).is_file()
        for staged_path in public.data_files
    )
    assert public.data_files == [f"inputs/{path}" for path in task.public.data_files]

    candidate = builder.compute_reference(TASK_ROOT / "public")
    grade = grade_candidate(
        candidate,
        PrivateGradingSpec(**task.private.model_dump()),
    )
    assert grade.passed and grade.score == 1.0

    results = candidate["key_results"]
    alpha = results["study_between_arm_comparisons"]["alpha"]
    beta = results["study_between_arm_comparisons"]["beta"]
    assert results["study_attrition"]["alpha"]["complete_pairs"] == 6
    assert results["study_attrition"]["beta"]["complete_pairs"] == 6
    assert alpha["difference_in_mean_change_b_minus_a"] == -5.0
    assert beta["difference_in_mean_change_b_minus_a"] == -3.0
    pooled = results["pooled_comparison"]
    pooled_value = pooled["pooled_difference_in_mean_change_b_minus_a"]
    assert pooled_value == -4.6
    simple_average = (
        alpha["difference_in_mean_change_b_minus_a"]
        + beta["difference_in_mean_change_b_minus_a"]
    ) / 2
    assert simple_average == -4.0


def test_code_architectures_receive_identical_public_context_and_paths(
    tmp_path: Path,
) -> None:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, builder.TASK_ID)
    public = stage_public_task(task.public, tmp_path / "attempt")
    staged = _resolved_public_files(public, tmp_path / "attempt")
    context = build_public_analysis_context(public, staged)
    serialized = json.dumps(context, ensure_ascii=False)

    one_shot = build_one_shot_code_messages(public, context)[1]["content"]
    single = build_single_agent_messages(public, tmp_path / "single", staged, context)[
        1
    ]["content"]
    planner_messages = build_planner_messages(
        question=public.prompt,
        input_context=serialized,
        replan_count=0,
        answer_schema=public.answer_schema,
        max_plan_goals=6,
    )
    planner = planner_messages[1]["content"]

    for prompt in (one_shot, single, planner):
        assert serialized in prompt
        assert "Study Alpha Amendment 01" in prompt
        assert "Platform `PY`" in prompt
        assert all(str(path.resolve()) in prompt for path in staged)

    assert "complete cohort as a declared" in planner_messages[0]["content"]
    assert "sole cohort source" in planner_messages[0]["content"]
    assert "single\ncohort-producing goal may apply" in planner_messages[0]["content"]

    private_reference = Path(task.private.reference_path).read_text(encoding="utf-8")
    assert private_reference not in serialized
    assert "private/reference.json" not in serialized
    assert "private/grader.py" not in serialized


@pytest.mark.parametrize(
    "mutate",
    [
        lambda result: next(
            row
            for row in result["key_results"]["selected_pairs"]
            if row["analysis_subject_id"] == "ALPHA-002"
        ).update(followup_visit_record_id="A02-FU-LEGACY"),
        lambda result: next(
            row
            for row in result["key_results"]["selected_pairs"]
            if row["analysis_subject_id"] == "ALPHA-004"
        ).update(baseline_harmonized_value=100.0),
        lambda result: next(
            row
            for row in result["key_results"]["selected_pairs"]
            if row["analysis_subject_id"] == "BETA-001"
        ).update(analysis_subject_id="BETA-002"),
        lambda result: next(
            row
            for row in result["key_results"]["selected_pairs"]
            if row["analysis_subject_id"] == "BETA-004"
        ).update(baseline_assay_record_ids=["B04-ASSAY-BL-R1"]),
        lambda result: result["key_results"]["pooled_comparison"].update(
            pooled_difference_in_mean_change_b_minus_a=-4.0
        ),
        lambda result: result["key_results"]["study_statistics"]["alpha"]["A"].update(
            sample_se_change=result["key_results"]["study_statistics"]["alpha"]["A"][
                "sample_sd_change"
            ]
        ),
        lambda result: result["key_results"]["study_between_arm_comparisons"][
            "alpha"
        ].update(difference_in_mean_change_b_minus_a=5.0),
        lambda result: result["key_results"]["study_attrition"]["beta"].update(
            complete_pairs=7
        ),
    ],
)
def test_private_grader_rejects_adversarial_scientific_mutations(mutate) -> None:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, builder.TASK_ID)
    candidate = copy.deepcopy(builder.compute_reference(TASK_ROOT / "public"))
    mutate(candidate)

    grade = grade_candidate(
        candidate,
        PrivateGradingSpec(**task.private.model_dump()),
    )

    assert not grade.passed


@pytest.mark.parametrize(
    "mutate",
    [
        lambda result: result["key_results"]["selected_pairs"].pop(0),
        lambda result: result["key_results"]["study_statistics"]["alpha"]["A"].update(
            mean_change=999.0
        ),
        lambda result: result["key_results"]["study_between_arm_comparisons"][
            "beta"
        ].update(variance_of_difference=999.0),
        lambda result: result["key_results"]["pooled_comparison"].update(
            sum_of_inverse_variance_weights=999.0
        ),
    ],
)
def test_cross_study_internal_consistency_guard_rejects_mixed_results(mutate) -> None:
    candidate = copy.deepcopy(builder.compute_reference(TASK_ROOT / "public"))
    assert _cross_study_consistency_error(candidate) is None

    mutate(candidate)

    assert _cross_study_consistency_error(candidate) is not None
