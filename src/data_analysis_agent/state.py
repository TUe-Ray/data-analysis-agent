"""Minimal shared state for the Prototype V0 graph."""

from typing import Literal, TypedDict


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
    final_answer: str
    trace: list[str]
