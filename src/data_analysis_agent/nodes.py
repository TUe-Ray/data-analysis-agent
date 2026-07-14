"""Node factories for the minimal verification-first graph."""

from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import ValidationError

from data_analysis_agent.final_output import (
    FinalGenerationRequest,
    FinalOutputProvider,
    OutputRepairRequest,
)
from data_analysis_agent.models import RoleModel
from data_analysis_agent.prompts import (
    VERIFIER_REPAIR_PROMPT,
    build_executor_messages,
    build_planner_messages,
    build_verifier_messages,
)
from data_analysis_agent.schemas import (
    FinalAnswer,
    FinalFailureAnswer,
    VerificationOutput,
)
from data_analysis_agent.state import (
    AgentState,
    IterationRecord,
    OutputValidationRecord,
)

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
                route = (
                    "Verifier -> Planner"
                    if output.decision == "REPLAN"
                    and state.get("replan_count", 0) < state.get("max_replans", 1)
                    else (
                        "Verifier -> Final Answer Generator"
                        if output.decision == "PASS"
                        else "Verifier -> Failure Finalizer"
                    )
                )
                record: IterationRecord = {
                    "iteration": state.get("replan_count", 0) + 1,
                    "plan": state["plan"],
                    "execution_result": state["execution_result"],
                    "verification_decision": output.decision,
                    "verification_feedback": output.feedback,
                    "route": route,
                }
                return {
                    "verification_decision": output.decision,
                    "verification_feedback": output.feedback,
                    "trace": _trace(state, f"verifier:{output.decision}"),
                    "iteration_history": [
                        *state.get("iteration_history", []),
                        record,
                    ],
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


def make_final_answer_generator_node(provider: FinalOutputProvider) -> Node:
    """Generate candidate JSON using only approved, public workflow values."""

    def final_answer_generator(state: AgentState) -> dict[str, object]:
        request = FinalGenerationRequest(
            question=state["question"],
            approved_execution_result=state["execution_result"],
            verifier_decision=state["verification_decision"],
            verifier_feedback=state["verification_feedback"],
            iteration_history=list(state.get("iteration_history", [])),
        )
        return {
            "raw_final_output": provider.generate(request),
            "output_repair_count": state.get("output_repair_count", 0),
            "max_output_repairs": state.get("max_output_repairs", 1),
            "trace": _trace(state, "final_answer_generator"),
        }

    return final_answer_generator


def output_validator_node(state: AgentState) -> dict[str, object]:
    """Validate JSON syntax and schema only, never scientific correctness."""
    raw_output = (
        state["raw_repair_output"]
        if state.get("output_repair_count", 0) > 0
        else state["raw_final_output"]
    )
    attempt = len(state.get("output_validation_history", [])) + 1
    try:
        parsed = json.loads(raw_output)
        validated = FinalAnswer.model_validate(parsed)
        validated_data = validated.model_dump(mode="json")
        final_answer = json.dumps(
            validated_data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as error:
        validation_error = f"{type(error).__name__}: {error}"
        repair_count = state.get("output_repair_count", 0)
        max_repairs = state.get("max_output_repairs", 1)
        route = (
            "Output Validator -> Output Repair"
            if repair_count < max_repairs
            else "Output Validator -> Output Failure"
        )
        record: OutputValidationRecord = {
            "attempt": attempt,
            "status": "INVALID",
            "error": validation_error,
            "route": route,
        }
        return {
            "validated_final_answer": None,
            "output_validation_status": "INVALID",
            "output_validation_error": validation_error,
            "output_validation_history": [
                *state.get("output_validation_history", []),
                record,
            ],
            "trace": _trace(state, "output_validator:INVALID"),
        }

    record = {
        "attempt": attempt,
        "status": "VALID",
        "error": "",
        "route": "Output Validator -> END",
    }
    return {
        "validated_final_answer": validated_data,
        "output_validation_status": "VALID",
        "output_validation_error": "",
        "output_validation_history": [
            *state.get("output_validation_history", []),
            record,
        ],
        "final_answer": final_answer,
        "status": validated.status,
        "final_status": validated.status,
        "trace": _trace(state, "output_validator:VALID"),
    }


def make_output_repair_node(provider: FinalOutputProvider) -> Node:
    """Create one formatting-only repair node with an intentionally narrow input."""

    def output_repair(state: AgentState) -> dict[str, object]:
        request = OutputRepairRequest(
            invalid_raw_output=state["raw_final_output"],
            validation_error=state["output_validation_error"],
            required_schema=FinalAnswer.model_json_schema(),
            approved_execution_result=state["execution_result"],
        )
        raw_repair_output = provider.repair(request)
        return {
            "raw_repair_output": raw_repair_output,
            "output_repair_count": state.get("output_repair_count", 0) + 1,
            "trace": _trace(state, "output_repair"),
        }

    return output_repair


def max_replan_failure_node(state: AgentState) -> dict[str, object]:
    """Return explicit failure JSON when scientific verification never passes."""
    feedback = state.get("verification_feedback", "No verifier feedback provided.")
    failure = FinalFailureAnswer(
        status="stopped_after_max_replans",
        answer=None,
        key_results={},
        limitations=["The latest execution result was not approved by the Verifier."],
        error=(
            "Verification did not pass before the maximum replan count was reached. "
            f"Latest verifier feedback: {feedback}"
        ),
    )
    return {
        "final_answer": failure.model_dump_json(indent=2),
        "validated_final_answer": failure.model_dump(mode="json"),
        "status": "stopped_after_max_replans",
        "final_status": "stopped_after_max_replans",
        "trace": _trace(state, "failure_finalizer:max_replans"),
    }


def output_failure_node(state: AgentState) -> dict[str, object]:
    """Terminate after the single repair allowance without claiming completion."""
    error = state.get("output_validation_error", "Unknown output validation error")
    failure = FinalFailureAnswer(
        status="output_validation_failed",
        answer=None,
        key_results={},
        limitations=["The approved result could not be formatted as valid JSON."],
        error=error,
    )
    return {
        "final_answer": failure.model_dump_json(indent=2),
        "validated_final_answer": failure.model_dump(mode="json"),
        "status": "output_validation_failed",
        "final_status": "output_validation_failed",
        "trace": _trace(state, "output_failure"),
    }
