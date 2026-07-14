"""LangGraph assembly and conditional routing for Prototype V0."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from data_analysis_agent.models import RoleModel
from data_analysis_agent.nodes import (
    finalize_node,
    make_executor_node,
    make_planner_node,
    make_verifier_node,
)
from data_analysis_agent.state import AgentState


def route_after_verification(state: AgentState) -> Literal["planner", "finalize"]:
    """Route validated decisions while enforcing the configured replan bound."""
    decision = state.get("verification_decision")
    if decision == "PASS":
        return "finalize"
    if decision == "REPLAN":
        if state.get("replan_count", 0) < state.get("max_replans", 1):
            return "planner"
        return "finalize"
    raise RuntimeError("Verifier did not provide a routing decision")


def build_graph(model: RoleModel):
    """Compile the minimal Planner -> Executor -> Verifier workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", make_planner_node(model))
    workflow.add_node("executor", make_executor_node(model))
    workflow.add_node("verifier", make_verifier_node(model))
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "verifier")
    workflow.add_conditional_edges(
        "verifier",
        route_after_verification,
        {"planner": "planner", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)
    return workflow.compile()
