"""LangGraph assembly and conditional routing for Prototype V0."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from data_analysis_agent.final_output import (
    DeterministicFinalOutputProvider,
    FinalOutputProvider,
)
from data_analysis_agent.models import RoleModel
from data_analysis_agent.nodes import (
    make_executor_node,
    make_final_answer_generator_node,
    make_output_repair_node,
    make_planner_node,
    make_verifier_node,
    max_replan_failure_node,
    output_failure_node,
    output_validator_node,
)
from data_analysis_agent.state import AgentState


def route_after_verification(
    state: AgentState,
) -> Literal["planner", "final_answer_generator", "failure_finalizer"]:
    """Route validated decisions while enforcing the configured replan bound."""
    decision = state.get("verification_decision")
    if decision == "PASS":
        return "final_answer_generator"
    if decision == "REPLAN":
        if state.get("replan_count", 0) < state.get("max_replans", 1):
            return "planner"
        return "failure_finalizer"
    raise RuntimeError("Verifier did not provide a routing decision")


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
):
    """Compile the bounded scientific and JSON-output validation workflow."""
    output_provider = output_provider or DeterministicFinalOutputProvider()
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", make_planner_node(model))
    workflow.add_node("executor", make_executor_node(model))
    workflow.add_node("verifier", make_verifier_node(model))
    workflow.add_node(
        "final_answer_generator",
        make_final_answer_generator_node(output_provider),
    )
    workflow.add_node("output_validator", output_validator_node)
    workflow.add_node("output_repair", make_output_repair_node(output_provider))
    workflow.add_node("failure_finalizer", max_replan_failure_node)
    workflow.add_node("output_failure", output_failure_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "verifier")
    workflow.add_conditional_edges(
        "verifier",
        route_after_verification,
        {
            "planner": "planner",
            "final_answer_generator": "final_answer_generator",
            "failure_finalizer": "failure_finalizer",
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
    workflow.add_edge("output_failure", END)
    return workflow.compile()
