"""Shared state for the bounded goal-driven analysis graph."""

from typing import Literal, TypedDict

from pydantic import JsonValue


class IterationRecord(TypedDict, total=False):
    """Public outputs from one Planner/Executor/Verifier iteration."""

    iteration: int
    plan: str
    execution_result: str
    verification_decision: Literal["PASS", "REPLAN"]
    verification_feedback: str
    route: str
    goal_id: str
    strategy: str
    capability_name: str | None


class OutputValidationRecord(TypedDict):
    """One schema-validation attempt for readable output and detailed logs."""

    attempt: int
    status: Literal["VALID", "INVALID"]
    error: str
    route: str


class CodeExecutionRecord(TypedDict, total=False):
    """One generated-code attempt, including the machine-readable failure type."""

    goal_id: str
    version: int
    attempt: int
    failure_category: str | None
    normalized_failure_family: str | None
    consecutive_failure_family_count: int
    exit_code: int | None
    timed_out: bool
    policy_validated: bool
    parsed_result: bool
    error: str | None
    source_changed: bool
    materially_changed: bool
    deterministic_result_recovery_attempted: bool
    route: str
    code_repair_attempts_for_current_goal: int
    scientific_replan_count: int


class PlannerValidationRecord(TypedDict, total=False):
    """One persisted raw Planner response and deterministic validation result."""

    mode: Literal["initial", "scientific_replan"]
    version: int
    raw_response_path: str
    validation_path: str
    valid: bool
    error_type: str | None
    error: str | None
    planner_repair_count: int
    scientific_replan_count: int
    route: str


class AgentState(TypedDict, total=False):
    """Values passed between the V0 Planner, Executor, and Verifier nodes."""

    question: str
    file_paths: list[str]
    input_context: str
    plan: str
    high_level_plan: dict[str, JsonValue]
    structured_plan: bool
    current_goal_index: int
    current_goal: dict[str, JsonValue]
    current_strategy: dict[str, JsonValue]
    current_goal_result: dict[str, JsonValue]
    completed_goal_results: list[dict[str, JsonValue]]
    pending_goal_artifacts: list[dict[str, JsonValue]]
    approved_goal_artifacts: list[dict[str, JsonValue]]
    approved_goal_artifacts_path: str
    capability_catalog: list[dict[str, JsonValue]]
    staged_file_paths: list[str]
    staged_file_display_paths: list[str]
    execution_working_directory: str
    executor_warnings: list[str]
    policy_failure_reason: str | None
    run_id: str
    run_directory: str
    trusted_tool_calls: int
    generated_script_count: int
    code_repair_count: int
    code_repair_attempts_for_current_goal: int
    max_code_repair_attempts: int
    code_repair_no_progress_count: int
    max_code_repair_no_progress_attempts: int
    code_repair_no_progress: bool
    consecutive_failure_family: str | None
    current_generated_code: str
    generated_execution_history: list[dict[str, JsonValue]]
    python_response_history: list[dict[str, JsonValue]]
    code_execution_history: list[CodeExecutionRecord]
    execution_failure_category: str | None
    failure_category: (
        Literal[
            "policy_error",
            "syntax_error",
            "runtime_error",
            "timeout",
            "result_contract_error",
            "generation_contract_error",
            "scientific_verification_failure",
        ]
        | None
    )
    execution_result: str
    verification_decision: Literal["PASS", "REPLAN"]
    verification_feedback: str
    replan_count: int
    max_replans: int
    stop_after_goals: int | None
    partial_run_reached: bool
    planner_mode: Literal["initial", "scientific_replan"]
    planner_repair_count: int
    max_planner_repairs: int
    planner_validation_error: str | None
    planner_raw_response: str
    planner_response_history: list[dict[str, JsonValue]]
    planner_validation_history: list[PlannerValidationRecord]
    planner_raw_response_path: str
    status: str
    final_status: str
    final_answer: str
    raw_final_output: str
    raw_repair_output: str
    validated_final_answer: dict[str, JsonValue] | None
    output_validation_status: Literal["VALID", "INVALID"] | None
    output_validation_error: str
    output_repair_count: int
    max_output_repairs: int
    output_validation_history: list[OutputValidationRecord]
    trace: list[str]
    iteration_history: list[IterationRecord]
