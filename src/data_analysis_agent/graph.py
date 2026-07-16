"""LangGraph assembly and conditional routing for Prototype V0."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from data_analysis_agent.final_output import (
    DeterministicFinalOutputProvider,
    FinalOutputProvider,
)
from data_analysis_agent.models import RoleModel
from data_analysis_agent.nodes import (
    make_executor_node,
    make_final_answer_generator_node,
    make_mechanical_repair_node,
    make_output_repair_node,
    make_planner_node,
    make_planner_repair_node,
    make_verifier_node,
    max_replan_failure_node,
    mechanical_execution_failure_node,
    output_failure_node,
    output_validator_node,
    partial_run_finalizer_node,
    planner_output_failure_node,
    planner_validator_node,
    select_current_goal_node,
)
from data_analysis_agent.python_runner import LocalPythonRunner
from data_analysis_agent.schemas import HighLevelPlan
from data_analysis_agent.state import AgentState

InputAdapter = Callable[[AgentState], dict[str, object]]


def route_after_verification(
    state: AgentState,
) -> Literal[
    "planner",
    "select_current_goal",
    "final_answer_generator",
    "failure_finalizer",
    "partial_run_finalizer",
]:
    """Route validated decisions while enforcing the configured replan bound."""
    decision = state.get("verification_decision")
    if decision == "PASS":
        target = state.get("stop_after_goals")
        if target and len(state.get("completed_goal_results", [])) >= target:
            return "partial_run_finalizer"
        if state.get("high_level_plan") is not None:
            plan = HighLevelPlan.model_validate(state["high_level_plan"])
            if state.get("current_goal_index", 0) < len(plan.goals):
                return "select_current_goal"
        return "final_answer_generator"
    if decision == "REPLAN":
        if state.get("replan_count", 0) < state.get("max_replans", 1):
            return "planner"
        return "failure_finalizer"
    raise RuntimeError("Verifier did not provide a routing decision")


def route_after_execution(
    state: AgentState,
) -> Literal["verifier", "mechanical_repair", "mechanical_failure"]:
    """Only successful generated executions may reach scientific verification."""
    strategy = state.get("current_strategy", {}).get("strategy")
    if strategy != "generated_python":
        return "verifier"
    result = state.get("current_goal_result")
    if result and result.get("success"):
        return "verifier"
    if state.get("code_repair_no_progress"):
        return "mechanical_failure"
    if state.get("code_repair_attempts_for_current_goal", 0) < state.get(
        "max_code_repair_attempts", 50
    ):
        return "mechanical_repair"
    return "mechanical_failure"


def route_after_planner_validation(
    state: AgentState,
) -> Literal["select_current_goal", "planner_repair", "planner_output_failure"]:
    """Route deterministic Planner schema failures to bounded structural repair."""
    if state.get("planner_validation_error") is None:
        return "select_current_goal"
    if state.get("planner_repair_count", 0) < state.get("max_planner_repairs", 2):
        return "planner_repair"
    return "planner_output_failure"


def route_after_output_validation(
    state: AgentState,
) -> Literal["end", "output_repair", "output_failure"]:
    """Route validated output status with a bounded formatting-only repair."""
    status = state.get("output_validation_status")
    if status == "VALID":
        return "end"
    if status == "INVALID":
        if state.get("output_repair_count", 0) < state.get("max_output_repairs", 1):
            return "output_repair"
        return "output_failure"
    raise RuntimeError("Output Validator did not provide a routing status")


def build_graph(
    model: RoleModel,
    output_provider: FinalOutputProvider | None = None,
    runner: LocalPythonRunner | None = None,
    *,
    input_schema: type[Any] | None = None,
    input_adapter: InputAdapter | None = None,
):
    """Compile the bounded scientific and JSON-output validation workflow.

    ``input_schema`` and ``input_adapter`` are an opt-in public-input boundary.
    They let a caller such as Studio expose a small form while retaining the full
    internal ``AgentState`` used by the existing workflow and its callers.
    """
    if (input_schema is None) != (input_adapter is None):
        raise ValueError("input_schema and input_adapter must be supplied together")
    output_provider = output_provider or DeterministicFinalOutputProvider()
    workflow = StateGraph(AgentState, input_schema=input_schema)
    if input_adapter is not None:
        workflow.add_node("prepare_input", input_adapter)
    workflow.add_node("planner", make_planner_node(model))
    workflow.add_node("planner_validator", planner_validator_node)
    workflow.add_node("planner_repair", make_planner_repair_node(model))
    workflow.add_node("select_current_goal", select_current_goal_node)
    workflow.add_node("executor", make_executor_node(model, runner))
    workflow.add_node("mechanical_repair", make_mechanical_repair_node(model, runner))
    workflow.add_node("verifier", make_verifier_node(model))
    workflow.add_node(
        "final_answer_generator",
        make_final_answer_generator_node(output_provider),
    )
    workflow.add_node("output_validator", output_validator_node)
    workflow.add_node("output_repair", make_output_repair_node(output_provider))
    workflow.add_node("failure_finalizer", max_replan_failure_node)
    workflow.add_node("planner_output_failure", planner_output_failure_node)
    workflow.add_node("mechanical_failure", mechanical_execution_failure_node)
    workflow.add_node("output_failure", output_failure_node)
    workflow.add_node("partial_run_finalizer", partial_run_finalizer_node)

    if input_adapter is None:
        workflow.add_edge(START, "planner")
    else:
        workflow.add_edge(START, "prepare_input")
        workflow.add_edge("prepare_input", "planner")
    workflow.add_edge("planner", "planner_validator")
    workflow.add_conditional_edges(
        "planner_validator",
        route_after_planner_validation,
        {
            "select_current_goal": "select_current_goal",
            "planner_repair": "planner_repair",
            "planner_output_failure": "planner_output_failure",
        },
    )
    workflow.add_edge("planner_repair", "planner_validator")
    workflow.add_edge("select_current_goal", "executor")
    workflow.add_conditional_edges(
        "executor",
        route_after_execution,
        {
            "verifier": "verifier",
            "mechanical_repair": "mechanical_repair",
            "mechanical_failure": "mechanical_failure",
        },
    )
    workflow.add_conditional_edges(
        "mechanical_repair",
        route_after_execution,
        {
            "verifier": "verifier",
            "mechanical_repair": "mechanical_repair",
            "mechanical_failure": "mechanical_failure",
        },
    )
    workflow.add_conditional_edges(
        "verifier",
        route_after_verification,
        {
            "planner": "planner",
            "select_current_goal": "select_current_goal",
            "final_answer_generator": "final_answer_generator",
            "failure_finalizer": "failure_finalizer",
            "partial_run_finalizer": "partial_run_finalizer",
        },
    )
    workflow.add_edge("final_answer_generator", "output_validator")
    workflow.add_conditional_edges(
        "output_validator",
        route_after_output_validation,
        {
            "end": END,
            "output_repair": "output_repair",
            "output_failure": "output_failure",
        },
    )
    workflow.add_edge("output_repair", "output_validator")
    workflow.add_edge("failure_finalizer", END)
    workflow.add_edge("planner_output_failure", END)
    workflow.add_edge("mechanical_failure", END)
    workflow.add_edge("output_failure", END)
    workflow.add_edge("partial_run_finalizer", END)
    return workflow.compile()
