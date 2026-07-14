"""Minimal shared state for the Prototype V0 graph."""

from typing import Literal, TypedDict

from pydantic import JsonValue


class IterationRecord(TypedDict):
    """Public outputs from one Planner/Executor/Verifier iteration."""

    iteration: int
    plan: str
    execution_result: str
    verification_decision: Literal["PASS", "REPLAN"]
    verification_feedback: str
    route: str


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
