"""Verified document reconciliation and explicit GoalResult propagation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_analysis_agent.benchmark import DEFAULT_TASKS_ROOT
from data_analysis_agent.benchmark_approaches import _resolved_public_files
from data_analysis_agent.benchmark_context import build_public_analysis_context
from data_analysis_agent.benchmark_tasks import load_benchmark_task, stage_public_task
from data_analysis_agent.models import ScriptedRoleModel
from data_analysis_agent.nodes import (
    PlannerOutputError,
    _attrition_consistency_error,
    _dependency_goal_results,
    _goal_input_context,
    _is_json_trailing_zero_only_replan,
    _reconciliation_contract_error,
    _result_limit_truncation_error,
    _upstream_attrition_conflict_error,
    _validate_document_reconciliation_plan,
    _validate_plan,
    make_verifier_node,
)
from data_analysis_agent.prompts import (
    VERIFIER_SYSTEM_PROMPT,
    build_python_generation_messages,
    build_verifier_messages,
)
from data_analysis_agent.python_runner import PythonExecutionResult
from data_analysis_agent.schemas import VerificationOutput

TASK_ID = "longitudinal_treatment_response_distributed"


def _plan() -> dict[str, object]:
    goals = [
        {
            "goal_id": "rules",
            "objective": "Reconcile effective rules, mappings, and identifiers.",
            "required_outputs": [
                "effective_specification",
                "field_mappings",
                "document_precedence",
            ],
            "constraints": ["Apply amendments over conflicting base sections."],
            "success_criteria": ["The compact contract covers every governing rule."],
            "depends_on": [],
        },
        {
            "goal_id": "cohort",
            "objective": "Normalize subjects and apply basic eligibility.",
            "required_outputs": ["cohort artifact"],
            "constraints": ["Use the verified effective rules."],
            "success_criteria": ["Cohort is normalized."],
            "depends_on": ["rules"],
        },
        {
            "goal_id": "visits",
            "objective": "Join, deduplicate, and validate visits.",
            "required_outputs": ["valid visits artifact"],
            "constraints": ["Use verified mappings."],
            "success_criteria": ["Visit records are valid."],
            "depends_on": ["cohort"],
        },
        {
            "goal_id": "selection",
            "objective": "Select baseline and follow-up and apply exclusions.",
            "required_outputs": ["selected pairs artifact"],
            "constraints": ["Use the amended follow-up window."],
            "success_criteria": ["Selections follow the effective contract."],
            "depends_on": ["visits"],
        },
        {
            "goal_id": "statistics",
            "objective": "Compute final statistics and attrition.",
            "required_outputs": ["statistics", "attrition", "selected pairs"],
            "constraints": ["Use the verified statistical definitions."],
            "success_criteria": ["Reported values are internally consistent."],
            "depends_on": ["selection"],
        },
        {
            "goal_id": "final",
            "objective": "Assemble the exact final JSON answer.",
            "required_outputs": ["status", "answer", "key_results", "limitations"],
            "constraints": ["Match the public answer schema exactly."],
            "success_criteria": ["The complete answer validates."],
            "depends_on": ["statistics"],
        },
    ]
    return {
        "scientific_objective": "Apply the governing protocol and analyze response.",
        "goals": goals,
        "final_output_goal_id": "final",
        "invalidate_from_goal_id": None,
    }


def _effective_result() -> dict[str, object]:
    return {
        "goal_id": "rules",
        "success": True,
        "strategy": "generated_python",
        "capability_name": None,
        "result": {
            "effective_specification": {
                "accepted_measurement_statuses": [
                    "STATUS_VALID",
                    "STATUS_REVIEWED",
                ],
                "accepted_status_meanings": ["valid", "reviewed"],
                "baseline_window_start_day": -14,
                "baseline_window_end_day": -1,
                "followup_window_start_day": 28,
                "followup_window_end_day": 42,
                "followup_target_day": 35,
                "post_start_exclusion_lower_bound": "strict",
                "post_start_exclusion_upper_bound": "inclusive",
                "sample_sd_denominator": "n-1",
                "sample_se_definition": "sample_sd / sqrt(n)",
                "between_arm_direction": "B-minus-A",
            },
            "field_mappings": {
                "canonical_subject_id": "analysis_subject_id",
                "visit_join_key": "encounter_key",
                "subject_source_key": "source_subject_key",
                "subjects_primary_key": "subject_key",
                "source_record_id": "source_record_id",
                "logical_duplicate_fields": [
                    "source_record_id",
                    "analysis_subject_id",
                    "observed_at",
                    "response_value",
                    "measurement_status_code",
                    "qc_score",
                    "origin_system_code",
                ],
            },
            "document_precedence": [
                "protocol_amendment_01 overrides conflicting base sections"
            ],
        },
        "warnings": [],
        "error": None,
        "artifact_paths": [],
    }


def test_six_goal_reconciliation_plan_is_valid_and_final_is_transitive() -> None:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, TASK_ID)

    plan, structured = _validate_plan(
        json.dumps(_plan()), answer_schema=task.public.answer_schema, max_plan_goals=6
    )

    assert structured
    assert len(plan.goals) == 6
    assert plan.goals[0].goal_id == "rules"
    assert plan.goals[0].depends_on == []
    assert plan.goals[-1].depends_on == ["statistics"]


def test_document_plan_validation_rejects_an_incomplete_reconciliation_contract(
    tmp_path: Path,
) -> None:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, TASK_ID)
    attempt = tmp_path / "attempt"
    public = stage_public_task(task.public, attempt)
    staged = _resolved_public_files(public, attempt)
    context = build_public_analysis_context(public, staged)
    incomplete = _plan()
    incomplete["goals"][0]["required_outputs"] = ["effective_rules"]
    plan, _ = _validate_plan(
        json.dumps(incomplete),
        answer_schema=public.answer_schema,
        max_plan_goals=6,
    )

    with pytest.raises(PlannerOutputError, match="field mappings, document precedence"):
        _validate_document_reconciliation_plan({"input_profile": context}, plan)


def test_reconciliation_verifier_sees_documents_and_downstream_sees_contract(
    tmp_path: Path,
) -> None:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, TASK_ID)
    attempt = tmp_path / "attempt"
    public = stage_public_task(task.public, attempt)
    staged = _resolved_public_files(public, attempt)
    context = build_public_analysis_context(public, staged)
    base_state = {
        "question": public.prompt,
        "high_level_plan": _plan(),
        "input_profile": context,
        "input_context": json.dumps(context, ensure_ascii=False),
        "completed_goal_results": [],
    }
    reconciliation_state = {
        **base_state,
        "current_goal": _plan()["goals"][0],
    }
    verifier = build_verifier_messages(
        question=public.prompt,
        input_context=_goal_input_context(reconciliation_state),
        execution_result=json.dumps(_effective_result()),
        scientific_objective=str(_plan()["scientific_objective"]),
        current_goal=_plan()["goals"][0],
    )[1]["content"]

    assert "Where this amendment conflicts" in verifier
    assert "STATUS_REVIEWED" in verifier
    assert "analysis_subject_id" in verifier
    assert "day 28 through day 42" in verifier

    dependent_state = {
        **base_state,
        "current_goal": _plan()["goals"][1],
        "completed_goal_results": [_effective_result()],
    }
    independent_goal = dict(_plan()["goals"][1])
    independent_goal["depends_on"] = []
    independent_state = {
        **dependent_state,
        "current_goal": independent_goal,
    }

    dependent_results = _dependency_goal_results(dependent_state)
    assert (
        dependent_results[0]["result"]["effective_specification"][
            "followup_window_start_day"
        ]
        == 28
    )
    assert _dependency_goal_results(independent_state) == []
    assert "Where this amendment conflicts" not in _goal_input_context(dependent_state)
    assert "verified effective rule contract" in _goal_input_context(dependent_state)


def test_downstream_python_prompt_prevents_semantic_code_refilter_regression(
    tmp_path: Path,
) -> None:
    messages = build_python_generation_messages(
        current_goal=_plan()["goals"][3],
        staged_file_paths=[str(tmp_path / "valid_visits.csv")],
        completed_goal_results=[_effective_result()],
        goal_directory=str(tmp_path / "goal"),
        input_context=json.dumps({"specification_documents": [], "csv_profiles": []}),
        approved_artifacts=[
            {
                "producer_goal_id": "visits",
                "path": str(tmp_path / "valid_visits.csv"),
                "columns": ["measurement_status_code", "patient_id"],
            }
        ],
        verification_feedback=(
            "Use amended followup_target_day 35, not obsolete base day 38."
        ),
        previous_attempt_result={"effective_specification": {"target_day": 38}},
        previous_attempt_code="__agent_result__ = {'target_day': 38}",
    )

    system = " ".join(messages[0]["content"].split())
    assert "Do not reapply an upstream validity" in system
    assert "semantic labels where physical codes are stored" in system
    assert "exclude technical row keys" in system
    assert "Never call drop_duplicates without the explicit scientific identity" in (
        system
    )
    assert "deduplicate on the complete scientific identity" in system
    assert "duplicate count from joined rows minus deduplicated rows" in system
    assert "invalid count from deduplicated rows minus valid rows" in system
    assert "eligibility-filtered subject artifact" in system
    assert "never infer total input subjects from an eligibility-filtered" in system
    assert "source-system identifiers and canonical analysis identifiers distinct" in (
        system
    )
    user = messages[1]["content"]
    assert "Rejected result from the immediately preceding attempt" in user
    assert "Previous source from that rejected attempt" in user
    assert "__agent_result__ = {'target_day': 38}" in user
    assert '"target_day": 38' in user
    assert "never join a source identifier directly to a canonical patient" in system
    assert "Do not rely on whitespace-sensitive" in system
    assert "Use amended followup_target_day 35" in messages[1]["content"]


def test_reconciliation_contract_requires_document_derived_physical_mappings(
    tmp_path: Path,
) -> None:
    task = load_benchmark_task(DEFAULT_TASKS_ROOT, TASK_ID)
    attempt = tmp_path / "attempt"
    public = stage_public_task(task.public, attempt)
    context = build_public_analysis_context(
        public, _resolved_public_files(public, attempt)
    )
    incomplete = _effective_result()["result"]
    assert isinstance(incomplete, dict)
    effective = incomplete["effective_specification"]
    assert isinstance(effective, dict)
    effective["accepted_measurement_statuses"] = ["valid", "reviewed"]
    incomplete["field_mappings"] = {
        "measurement_status_codes": "protocol_amendment_01.md"
    }
    state = {
        "current_goal": _plan()["goals"][0],
        "input_profile": context,
    }

    error = _reconciliation_contract_error(state, incomplete)

    assert error is not None
    assert "STATUS_VALID" in error
    assert "STATUS_REVIEWED" in error
    assert "encounter_key" in error
    assert "analysis_subject_id" in error
    assert _reconciliation_contract_error(state, _effective_result()["result"]) is None

    invalid_codes = _effective_result()["result"]
    assert isinstance(invalid_codes, dict)
    invalid_effective = invalid_codes["effective_specification"]
    assert isinstance(invalid_effective, dict)
    invalid_effective["accepted_measurement_statuses"] = [
        "STATUS_VALID",
        "STATUS_REVIEWED",
        "STATUS_SENSOR_FAIL",
    ]
    invalid_error = _reconciliation_contract_error(state, invalid_codes)
    assert invalid_error is not None
    assert "STATUS_SENSOR_FAIL" in invalid_error


def test_obsolete_base_window_routes_to_replan_against_verified_contract() -> None:
    plan = _plan()
    current = dict(plan["goals"][3])
    current["constraints"] = ["Use the obsolete base day 21 through day 35 window."]
    state = {
        "question": "Analyze using the governing protocol.",
        "structured_plan": True,
        "high_level_plan": plan,
        "plan": json.dumps(plan),
        "current_goal": current,
        "current_goal_index": 3,
        "current_goal_result": {
            "goal_id": "selection",
            "success": True,
            "strategy": "generated_python",
            "capability_name": None,
            "result": {"window_used": [21, 35]},
            "warnings": [],
            "error": None,
            "artifact_paths": [],
        },
        "execution_result": '{"window_used":[21,35]}',
        "current_strategy": {
            "strategy": "generated_python",
            "capability_name": None,
            "arguments": {},
            "concise_reason": "Select visits.",
        },
        "completed_goal_results": [_effective_result()],
        "input_profile": {"specification_documents": [], "csv_profiles": []},
        "input_context": "",
        "replan_count": 0,
        "max_replans": 1,
        "trace": [],
    }
    model = ScriptedRoleModel(
        {
            "verifier": [
                '{"decision":"REPLAN","feedback":"Use verified amended days '
                '28 through 42, not obsolete days 21 through 35."}'
            ]
        }
    )

    update = make_verifier_node(model)(state)

    assert update["verification_decision"] == "REPLAN"
    prompt = model.calls[0].messages[1]["content"]
    assert '"followup_window_start_day": 28' in prompt
    assert "obsolete base day 21 through day 35" in prompt
    assert "Where this amendment conflicts" not in prompt


def test_dependency_free_result_error_routes_to_goal_retry() -> None:
    plan = _plan()
    state = {
        "question": "Reconcile the governing protocol.",
        "structured_plan": True,
        "high_level_plan": plan,
        "plan": json.dumps(plan),
        "current_goal": plan["goals"][0],
        "current_goal_index": 0,
        "current_goal_result": {
            "goal_id": "rules",
            "success": True,
            "strategy": "generated_python",
            "capability_name": None,
            "result": {
                "effective_specification": {"window": [21, 35]},
                "field_mappings": {"arm": {"A": "A"}},
                "document_precedence": ["base protocol"],
            },
            "warnings": [],
            "error": None,
            "artifact_paths": [],
        },
        "execution_result": "{}",
        "current_strategy": {
            "strategy": "generated_python",
            "capability_name": None,
            "arguments": {},
            "concise_reason": "Reconcile rules.",
        },
        "completed_goal_results": [],
        "input_profile": {"specification_documents": [], "csv_profiles": []},
        "input_context": "",
        "replan_count": 0,
        "max_replans": 1,
        "max_goal_retries": 2,
        "trace": [],
    }
    model = ScriptedRoleModel(
        {
            "verifier": [
                '{"decision":"REPLAN","issue_classification":"result",'
                '"feedback":"The arm mapping is incorrect."}'
            ]
        }
    )

    update = make_verifier_node(model)(state)

    assert update["verification_decision"] == "RETRY_GOAL"
    assert update["verification_feedback"] == "The arm mapping is incorrect."


def test_verifier_does_not_replan_for_json_trailing_zero_representation() -> None:
    normalized = " ".join(VERIFIER_SYSTEM_PROMPT.split())

    assert "0.34 and 0.340 are the same JSON numeric value" in normalized
    assert "Never request a scientific replan solely to display trailing zeros" in (
        normalized
    )


def test_deterministic_verifier_rejects_overlapping_attrition_counts() -> None:
    overlapping_counts = {
        "total_patients": 48,
        "basic_ineligible": 6,
        "eligible_after_basic_checks": 42,
        "excluded_pre_start": 21,
        "no_valid_baseline": 11,
        "no_valid_followup": 4,
        "excluded_post_start_before_or_on_followup": 11,
        "complete_pairs": 27,
        "complete_pairs_arm_a": 14,
        "complete_pairs_arm_b": 13,
    }
    overlapping = {
        "attrition_counts": overlapping_counts,
        "selected_pairs": [{}] * 27,
    }
    sequential = {
        "attrition_counts": {
            **overlapping_counts,
            "excluded_pre_start": 4,
            "no_valid_baseline": 5,
            "excluded_post_start_before_or_on_followup": 2,
        },
        "selected_pairs": [{}] * 27,
    }

    error = _attrition_consistency_error(overlapping)
    assert error is not None
    assert "must equal eligible_after_basic_checks (42), but equals 74" in error
    assert _attrition_consistency_error(sequential) is None
    negative = {
        **sequential,
        "attrition_counts": {
            **sequential["attrition_counts"],
            "no_valid_followup": -1,
        },
    }
    assert "counts must be nonnegative" in (
        _attrition_consistency_error(negative) or ""
    )


def test_verifier_rejects_attrition_that_conflicts_with_a_verified_dependency() -> None:
    plan = _plan()
    state = {
        "high_level_plan": plan,
        "current_goal": plan["goals"][3],
        "completed_goal_results": [
            {
                "goal_id": "visits",
                "success": True,
                "result": {
                    "attrition": {
                        "exact_duplicate_visit_rows_removed": 12,
                        "invalid_or_missing_visit_rows_excluded": 93,
                    }
                },
            }
        ],
    }
    result = {
        "attrition": {
            "exact_duplicate_visit_rows_removed": 0,
            "invalid_or_missing_visit_rows_excluded": 95,
        }
    }

    error = _upstream_attrition_conflict_error(state, result)

    assert error is not None
    assert "verified upstream goal visits" in error
    assert "exact_duplicate_visit_rows_removed is 0" in error
    assert "approved value is 12" in error


def test_result_limit_repair_rejects_silent_table_truncation() -> None:
    previous = PythonExecutionResult(
        success=False,
        version=1,
        exit_code=1,
        stdout="",
        stderr="",
        result={},
        error=(
            "ResultContractError: result list length 226 exceeds 100; store large "
            "tables as declared artifacts"
        ),
        duration_seconds=0.0,
        script_path="",
        failure_category="result_contract_error",
    )

    assert _result_limit_truncation_error(
        previous, "valid_visits = valid_visits[:100]\n"
    )
    assert _result_limit_truncation_error(
        previous, "valid_visits = valid_visits.head(100)\n"
    )
    assert (
        _result_limit_truncation_error(
            previous,
            "valid_visits.to_csv('valid_visits.csv', index=False)\n",
        )
        is None
    )


def test_python_generation_requires_operational_code_contract_and_artifact_handoff(
    tmp_path: Path,
) -> None:
    messages = build_python_generation_messages(
        current_goal={
            "goal_id": "rules",
            "objective": "Reconcile specifications and publish a table artifact.",
            "required_outputs": [
                "effective_specification",
                "field_mappings",
                "document_precedence",
                "valid visits artifact",
            ],
            "constraints": [],
            "success_criteria": [],
            "depends_on": [],
        },
        staged_file_paths=[],
        completed_goal_results=[],
        goal_directory=str(tmp_path),
    )

    system = messages[0]["content"]
    assert "exact physical values stored in data" in system
    assert "filter on physical" in system
    assert "values, not semantic labels" in system
    assert "scientific logical-duplicate identity" in system
    assert "explicitly marked not accepted" in system
    assert "merely writing the file is not enough" in system
    assert '__agent_result__["artifacts"]' in system


def test_json_trailing_zero_display_request_does_not_trigger_scientific_replan() -> (
    None
):
    display_only = VerificationOutput(
        decision="REPLAN",
        feedback=(
            "sample_se_change is 0.34 (two decimal places) and should be "
            "reported with exactly three decimal places as 0.340."
        ),
    )
    scientific_error = VerificationOutput(
        decision="REPLAN",
        feedback=(
            "sample_se_change has the incorrect value and does not match the "
            "sample SD / sqrt(n) formula at three decimal places."
        ),
    )

    assert _is_json_trailing_zero_only_replan(display_only)
    assert not _is_json_trailing_zero_only_replan(scientific_error)
