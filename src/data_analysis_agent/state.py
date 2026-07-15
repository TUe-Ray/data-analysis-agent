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
    capability_catalog: list[dict[str, JsonValue]]
    staged_file_paths: list[str]
    run_id: str
    run_directory: str
    trusted_tool_calls: int
    generated_script_count: int
    code_repair_count: int
    execution_result: str
    verification_decision: Literal["PASS", "REPLAN"]
    verification_feedback: str
    replan_count: int
    max_replans: int
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
