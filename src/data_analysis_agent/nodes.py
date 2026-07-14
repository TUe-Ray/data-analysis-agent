"""Node factories for the minimal verification-first graph."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError

from data_analysis_agent.models import RoleModel
from data_analysis_agent.prompts import (
    VERIFIER_REPAIR_PROMPT,
    build_executor_messages,
    build_planner_messages,
    build_verifier_messages,
)
from data_analysis_agent.schemas import VerificationOutput
from data_analysis_agent.state import AgentState

Node = Callable[[AgentState], dict[str, object]]


class VerifierOutputError(ValueError):
    """Raised when the Verifier cannot return valid structured output."""


def _trace(state: AgentState, event: str) -> list[str]:
    return [*state.get("trace", []), event]


def make_planner_node(model: RoleModel) -> Node:
    """Create a Planner node backed by the injected role model."""

    def planner(state: AgentState) -> dict[str, object]:
        is_replan = state.get("verification_decision") == "REPLAN"
        replan_count = state.get("replan_count", 0) + int(is_replan)
        feedback = state.get("verification_feedback") if is_replan else None
        messages = build_planner_messages(
            question=state["question"],
            input_context=state["input_context"],
            replan_count=replan_count,
            verification_feedback=feedback,
        )
        return {
            "plan": model.generate(role="planner", messages=messages),
            "replan_count": replan_count,
            "max_replans": state.get("max_replans", 1),
            "trace": _trace(state, "planner"),
        }

    return planner


def make_executor_node(model: RoleModel) -> Node:
    """Create an Executor node backed by the injected role model."""

    def executor(state: AgentState) -> dict[str, object]:
        messages = build_executor_messages(
            question=state["question"],
            input_context=state["input_context"],
            plan=state["plan"],
        )
        return {
            "execution_result": model.generate(role="executor", messages=messages),
            "trace": _trace(state, "executor"),
        }

    return executor


def make_verifier_node(model: RoleModel) -> Node:
    """Create a Verifier node with one bounded structured-output repair."""

    def verifier(state: AgentState) -> dict[str, object]:
        messages = build_verifier_messages(
            question=state["question"],
            input_context=state["input_context"],
            plan=state["plan"],
            execution_result=state["execution_result"],
        )
        validation_error: ValidationError | None = None
        for attempt in range(2):
            raw_output = model.generate(role="verifier", messages=messages)
            try:
                output = VerificationOutput.model_validate_json(raw_output)
                return {
                    "verification_decision": output.decision,
                    "verification_feedback": output.feedback,
                    "trace": _trace(state, f"verifier:{output.decision}"),
                }
            except ValidationError as error:
                validation_error = error
                if attempt == 0:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw_output},
                        {"role": "user", "content": VERIFIER_REPAIR_PROMPT},
                    ]

        raise VerifierOutputError(
            "Verifier returned invalid JSON after one repair attempt"
        ) from validation_error

    return verifier


def finalize_node(state: AgentState) -> dict[str, object]:
    """Finalize deterministically without another model request."""
    if state.get("verification_decision") == "PASS":
        return {
            "final_answer": state["execution_result"],
            "status": "completed",
            "trace": _trace(state, "finalize"),
        }

    feedback = state.get("verification_feedback", "No verifier feedback provided.")
    final_answer = (
        f"{state.get('execution_result', '')}\n\n"
        "Verification did not pass before the maximum replan count was reached. "
        f"Latest verifier feedback: {feedback}"
    )
    return {
        "final_answer": final_answer,
        "status": "stopped_after_max_replans",
        "trace": _trace(state, "finalize"),
    }
