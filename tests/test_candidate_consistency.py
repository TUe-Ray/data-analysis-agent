"""Bounded final-checker evidence without embedding raw scientific tables."""

from __future__ import annotations

import copy
import csv
import io
from pathlib import Path

from data_analysis_agent.benchmark import DEFAULT_TASKS_ROOT
from data_analysis_agent.benchmark_approaches import (
    _resolved_public_files,
    build_final_checker_messages,
)
from data_analysis_agent.benchmark_context import build_public_analysis_context
from data_analysis_agent.benchmark_tasks import load_benchmark_task, stage_public_task
from data_analysis_agent.candidate_consistency import (
    build_candidate_consistency_evidence,
)
from scripts.build_distributed_longitudinal_task import compute_oracle

TASK_ID = "longitudinal_treatment_response_distributed"


def test_candidate_consistency_evidence_accepts_oracle_identities() -> None:
    evidence = build_candidate_consistency_evidence(compute_oracle())

    assert evidence["label"] == "candidate-internal consistency evidence"
    assert evidence["pair_change_mismatches"] == []
    assert all(evidence["aggregate_checks"].values())
    assert all(evidence["attrition_identity_checks"].values())


def test_candidate_consistency_detects_required_contradiction_classes() -> None:
    candidate = compute_oracle()

    wrong_change = copy.deepcopy(candidate)
    wrong_change["key_results"]["selected_pairs"][0]["change"] += 1
    assert build_candidate_consistency_evidence(wrong_change)["pair_change_mismatches"]

    wrong_count = copy.deepcopy(candidate)
    wrong_count["key_results"]["attrition"]["complete_pairs_arm_a"] += 1
    assert not build_candidate_consistency_evidence(wrong_count)[
        "attrition_identity_checks"
    ]["complete_pair_arm_counts_match_selected_pairs"]

    wrong_mean = copy.deepcopy(candidate)
    wrong_mean["key_results"]["arm_statistics"]["A"]["mean_change"] += 1
    assert not build_candidate_consistency_evidence(wrong_mean)["aggregate_checks"][
        "arm_a_mean_change_matches_pairs"
    ]

    sd_se_confusion = copy.deepcopy(candidate)
    arm_a = sd_se_confusion["key_results"]["arm_statistics"]["A"]
    arm_a["sample_sd_change"], arm_a["sample_se_change"] = (
        arm_a["sample_se_change"],
        arm_a["sample_sd_change"],
    )
    evidence = build_candidate_consistency_evidence(sd_se_confusion)
    assert not evidence["aggregate_checks"]["arm_a_sample_sd_matches_pairs"]
    assert not evidence["aggregate_checks"]["arm_a_sample_se_matches_pairs"]

    wrong_direction = copy.deepcopy(candidate)
    comparison = wrong_direction["key_results"]["between_arm_comparison"]
    comparison["difference_in_mean_change_b_minus_a"] *= -1
    assert not build_candidate_consistency_evidence(wrong_direction)[
        "aggregate_checks"
    ]["b_minus_a_matches_pairs"]


def test_final_checker_gets_complete_documents_profiles_and_no_raw_table(
    tmp_path: Path,
) -> None:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, TASK_ID)
    attempt = tmp_path / "attempt"
    public = stage_public_task(task.public, attempt)
    staged = _resolved_public_files(public, attempt)
    context = build_public_analysis_context(public, staged)
    prompt = build_final_checker_messages(
        public=public,
        candidate=compute_oracle(),
        execution={"success": True, "exit_code": 0},
        analysis_context=context,
    )[1]["content"]

    assert "Where this amendment conflicts" in prompt
    assert "analysis_subject_id" in prompt
    assert "STATUS_REVIEWED" in prompt
    assert '"row_count": 331' in prompt
    assert "candidate-internal consistency evidence" in prompt
    measurements = list(
        csv.DictReader(
            io.StringIO(public.data_contents["inputs/data/measurements.csv"])
        )
    )
    assert measurements[3]["encounter_key"] not in prompt
    assert "BEGIN FILE: measurements.csv" not in prompt
