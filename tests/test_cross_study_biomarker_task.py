"""Static fairness, oracle, and strict-grader coverage for the cross-study task."""
# ruff: noqa: E501

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
from data_analysis_agent.task_builders import cross_study_biomarker as builder


def _grade(candidate: dict) -> object:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, builder.TASK_ID)
    return grade_candidate(candidate, PrivateGradingSpec(**task.private.model_dump()))


def test_builder_is_deterministic_oracle_valid_and_fully_static(tmp_path: Path) -> None:
    assert builder.generated_files() == builder.generated_files()
    builder.validate_task()
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, builder.TASK_ID)
    assert task.public.data_files == builder.PUBLIC_FILES
    assert task.public.deferred_public_files == {}
    assert task.public.release_stages == []
    staged = stage_public_task(task.public, tmp_path / "attempt")
    assert staged.data_files == [f"inputs/{name}" for name in builder.PUBLIC_FILES]
    assert _grade(builder.compute_oracle()).passed


def test_shared_context_is_equal_for_code_architectures_and_private_values_stay_private(
    tmp_path: Path,
) -> None:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, builder.TASK_ID)
    public = stage_public_task(task.public, tmp_path / "attempt")
    paths = _resolved_public_files(public, tmp_path / "attempt")
    context = build_public_analysis_context(public, paths)
    single = "\n".join(x["content"] for x in build_single_agent_messages(public, tmp_path, paths, context))
    one_shot = "\n".join(x["content"] for x in build_one_shot_code_messages(public, context))
    reference = Path(task.private.reference_path).read_text(encoding="utf-8")
    assert json.dumps(context) in single and json.dumps(context) in one_shot
    assert (
        public.data_contents["inputs/data/study_alpha_visits.csv"] not in one_shot
    )  # Complete raw tables are not in code prompts.
    assert reference not in single and reference not in one_shot
    assert "study_alpha_amendment_01" in single and "B_ACCEPT" in single


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x["key_results"]["selected_pairs"][0].update(followup_visit_record_id="A001-F-BOUND"),
        lambda x: x["key_results"]["selected_pairs"][0].update(followup_visit_record_id="A001-F-LEGACY"),
        lambda x: x["key_results"]["selected_pairs"][-1].update(followup_visit_record_id="B001-F-BAD"),
        lambda x: x["key_results"]["selected_pairs"][0].update(baseline_harmonized_value=0.0),
        lambda x: x["key_results"]["selected_pairs"][0].update(baseline_assay_record_ids=["AVERAGED"]),
        lambda x: x["key_results"]["selected_pairs"][-1].update(followup_assay_record_ids=["B001-F-R1"]),
        lambda x: x["key_results"]["selected_pairs"][-1].update(analysis_subject_id="BETA001"),
        lambda x: x["key_results"]["selected_pairs"][1].update(followup_visit_record_id="A002-F36-B"),
        lambda x: x["key_results"]["study_attrition"]["alpha"].update(excluded_post_start_before_followup=0),
        lambda x: x["key_results"]["study_statistics"]["alpha"]["A"].update(sample_se_change=x["key_results"]["study_statistics"]["alpha"]["A"]["sample_sd_change"]),
        lambda x: x["key_results"]["study_between_arm_comparisons"]["alpha"].update(difference_in_mean_change_b_minus_a=-x["key_results"]["study_between_arm_comparisons"]["alpha"]["difference_in_mean_change_b_minus_a"]),
        lambda x: x["key_results"]["pooled_comparison"].update(pooled_difference_in_mean_change_b_minus_a=sum(v["difference_in_mean_change_b_minus_a"] for v in x["key_results"]["study_between_arm_comparisons"].values()) / 2),
        lambda x: x["key_results"]["selected_pairs"][0].update(baseline_visit_record_id="wrong-record"),
        lambda x: x["key_results"]["study_attrition"]["beta"].update(complete_pairs=999),
    ],
)
def test_private_grader_rejects_required_scientific_mutations(mutate) -> None:
    candidate = copy.deepcopy(builder.compute_oracle())
    mutate(candidate)
    assert not _grade(candidate).passed
