"""Node factories for the bounded goal-driven verification-first graph."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path

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
    build_python_generation_messages,
    build_python_repair_messages,
    build_verifier_messages,
)
from data_analysis_agent.python_runner import (
    LocalPythonRunner,
    PythonExecutionResult,
    write_execution_metadata,
)
from data_analysis_agent.schemas import (
    ExecutionStrategy,
    FinalAnswer,
    FinalFailureAnswer,
    GoalResult,
    HighLevelPlan,
    IntermediateGoal,
    VerificationOutput,
)
from data_analysis_agent.state import AgentState, OutputValidationRecord
from data_analysis_agent.trusted_tools import TrustedToolRegistry

Node = Callable[[AgentState], dict[str, object]]


class PlannerOutputError(ValueError):
    """Raised when a JSON Planner response is not a valid HighLevelPlan."""


class ExecutorOutputError(ValueError):
    """Raised when the Executor does not return a valid strategy or source."""


class VerifierOutputError(ValueError):
    """Raised when the Verifier cannot return valid structured output."""


def _trace(state: AgentState, event: str) -> list[str]:
    return [*state.get("trace", []), event]


def _validate_plan(raw_output: str) -> tuple[HighLevelPlan, bool]:
    try:
        plan = HighLevelPlan.model_validate_json(raw_output)
    except ValidationError as error:
        if raw_output.lstrip().startswith("{"):
            raise PlannerOutputError(
                "Planner returned JSON that does not match HighLevelPlan"
            ) from error
        # Compatibility for the original V0 scripted scenarios. Live and new
        # offline flows are prompted to use the structured schema.
        return (
            HighLevelPlan(
                scientific_objective=raw_output.strip(),
                goals=[
                    IntermediateGoal(
                        goal_id="legacy_goal",
                        objective=raw_output.strip(),
                        required_outputs=[],
                        constraints=[],
                        success_criteria=["The supplied legacy plan is executed."],
                    )
                ],
            ),
            False,
        )
    goal_ids = [goal.goal_id for goal in plan.goals]
    if len(goal_ids) != len(set(goal_ids)):
        raise PlannerOutputError("Planner goal_id values must be unique")
    seen: set[str] = set()
    for goal in plan.goals:
        unknown = [
            dependency for dependency in goal.depends_on if dependency not in seen
        ]
        if unknown:
            raise PlannerOutputError(
                f"Goal {goal.goal_id!r} depends on a missing or later goal"
            )
        seen.add(goal.goal_id)
    return plan, True


def _preserve_completed_results(
    *,
    state: AgentState,
    revised_plan: HighLevelPlan,
) -> list[dict[str, object]]:
    previous = state.get("high_level_plan", {})
    previous_goals = {
        str(goal.get("goal_id")): str(goal.get("objective"))
        for goal in previous.get("goals", [])
        if isinstance(goal, dict)
    }
    revised_goals = {goal.goal_id: goal.objective for goal in revised_plan.goals}
    return [
        dict(result)
        for result in state.get("completed_goal_results", [])
        if (
            str(result.get("goal_id")) in revised_goals
            and previous_goals.get(str(result.get("goal_id")))
            == revised_goals[str(result.get("goal_id"))]
        )
    ]


def make_planner_node(model: RoleModel) -> Node:
    """Create a Planner that returns global, implementation-agnostic goals."""

    def planner(state: AgentState) -> dict[str, object]:
        is_replan = state.get("verification_decision") == "REPLAN"
        replan_count = state.get("replan_count", 0) + int(is_replan)
        messages = build_planner_messages(
            question=state["question"],
            input_context=state["input_context"],
            replan_count=replan_count,
            verification_feedback=(
                state.get("verification_feedback") if is_replan else None
            ),
            previous_plan=(state.get("high_level_plan") if is_replan else None),
            completed_goal_results=(
                state.get("completed_goal_results", []) if is_replan else None
            ),
            current_goal_failure=(
                state.get("current_goal_result") if is_replan else None
            ),
        )
        raw_plan = model.generate(role="planner", messages=messages)
        plan, structured = _validate_plan(raw_plan)
        completed = (
            _preserve_completed_results(state=state, revised_plan=plan)
            if is_replan and structured
            else ([] if is_replan else list(state.get("completed_goal_results", [])))
        )
        completed_ids = {str(item.get("goal_id")) for item in completed}
        next_index = next(
            (
                index
                for index, goal in enumerate(plan.goals)
                if goal.goal_id not in completed_ids
            ),
            len(plan.goals),
        )
        run_id = state.get("run_id") or uuid.uuid4().hex[:12]
        run_directory = state.get("run_directory") or str(
            Path.cwd() / "runs" / f"run_{run_id}"
        )
        return {
            "plan": raw_plan,
            "high_level_plan": plan.model_dump(mode="json"),
            "structured_plan": structured,
            "completed_goal_results": completed,
            "current_goal_index": next_index,
            "replan_count": replan_count,
            "max_replans": state.get("max_replans", 1),
            "run_id": run_id,
            "run_directory": run_directory,
            "trusted_tool_calls": state.get("trusted_tool_calls", 0),
            "generated_script_count": state.get("generated_script_count", 0),
            "code_repair_count": state.get("code_repair_count", 0),
            "trace": _trace(state, "planner"),
        }

    return planner


def select_current_goal_node(state: AgentState) -> dict[str, object]:
    """Select the next ordered goal without adding a scheduling architecture."""
    plan = HighLevelPlan.model_validate(state["high_level_plan"])
    index = state.get("current_goal_index", 0)
    if index >= len(plan.goals):
        raise RuntimeError("No remaining intermediate goal is available")
    goal = plan.goals[index]
    completed_ids = {
        str(item.get("goal_id")) for item in state.get("completed_goal_results", [])
    }
    missing = [item for item in goal.depends_on if item not in completed_ids]
    if missing:
        raise RuntimeError(
            f"Goal {goal.goal_id!r} has incomplete dependencies: {', '.join(missing)}"
        )
    return {"current_goal": goal.model_dump(mode="json")}


def _staged_paths(state: AgentState) -> list[Path]:
    supplied = state.get("staged_file_paths", state.get("file_paths", []))
    return [Path(path).resolve() for path in supplied if Path(path).is_file()]


def _staged_display_paths(state: AgentState) -> list[str]:
    supplied = state.get("staged_file_display_paths")
    if supplied is not None:
        return list(supplied)
    return [str(path) for path in _staged_paths(state)]


def _registry(state: AgentState) -> TrustedToolRegistry:
    staged = _staged_paths(state)
    roots = list(dict.fromkeys(path.parent for path in staged)) or [Path.cwd()]
    return TrustedToolRegistry(allowed_roots=roots, allowed_files=staged)


def _extract_code(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("code"), str):
            stripped = parsed["code"].strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if not stripped:
        raise ExecutorOutputError("Executor returned empty generated Python")
    return stripped + "\n"


def _generated_result(
    *,
    state: AgentState,
    model: RoleModel,
    strategy: ExecutionStrategy,
    runner: LocalPythonRunner,
) -> tuple[GoalResult, str, int, int]:
    goal = IntermediateGoal.model_validate(state["current_goal"])
    goal_directory = Path(state["run_directory"]) / "goals" / goal.goal_id
    staged_paths = _staged_paths(state)
    existing_versions = [
        int(path.stem.rsplit("v", 1)[-1])
        for path in goal_directory.glob("generated_code_v*.py")
        if path.stem.rsplit("v", 1)[-1].isdigit()
    ]
    first_version = max(existing_versions, default=0) + 1
    source_messages = build_python_generation_messages(
        current_goal=goal.model_dump(mode="json"),
        staged_file_paths=_staged_display_paths(state),
        completed_goal_results=list(state.get("completed_goal_results", [])),
        goal_directory=str(goal_directory),
    )
    code_v1 = _extract_code(model.generate(role="executor", messages=source_messages))
    executions: list[PythonExecutionResult] = [
        runner.run(
            code=code_v1,
            goal_directory=goal_directory,
            allowed_files=staged_paths,
            version=first_version,
            working_directory=(
                Path(state["execution_working_directory"])
                if state.get("execution_working_directory")
                else None
            ),
        )
    ]
    repair_count = 0
    if not executions[0].success:
        repair_messages = build_python_repair_messages(
            current_goal=goal.model_dump(mode="json"),
            code=code_v1,
            stdout=executions[0].stdout,
            stderr=executions[0].stderr,
            error=executions[0].error,
            staged_file_paths=_staged_display_paths(state),
        )
        code_v2 = _extract_code(
            model.generate(role="executor", messages=repair_messages)
        )
        executions.append(
            runner.run(
                code=code_v2,
                goal_directory=goal_directory,
                allowed_files=staged_paths,
                version=first_version + 1,
                working_directory=(
                    Path(state["execution_working_directory"])
                    if state.get("execution_working_directory")
                    else None
                ),
            )
        )
        repair_count = 1
    metadata_path = write_execution_metadata(
        goal_directory=goal_directory,
        run_id=state["run_id"],
        goal_id=goal.goal_id,
        executions=executions,
        strategy_reason=strategy.concise_reason,
    )
    final = executions[-1]
    artifact_paths = list(
        dict.fromkeys(
            [path for execution in executions for path in execution.artifact_paths]
            + [str(metadata_path)]
        )
    )
    saved_artifact_count = sum(path.is_file() for path in goal_directory.rglob("*"))
    latest_stderr_path = goal_directory / f"stderr_v{final.version}.txt"
    result = GoalResult(
        goal_id=goal.goal_id,
        success=final.success,
        strategy="generated_python",
        capability_name=None,
        result=final.result,
        warnings=[],
        error=final.error,
        artifact_paths=artifact_paths,
    )
    factual = {
        "success": final.success,
        "result": final.result,
        "exit_code": final.exit_code,
        "error": final.error,
        "duration_seconds": final.duration_seconds,
        "repair_required": repair_count == 1,
        "timed_out": final.timed_out,
        "artifact_count": saved_artifact_count,
        "artifact_directory": str(goal_directory),
        "latest_stderr_path": str(latest_stderr_path),
        "artifact_paths": artifact_paths,
    }
    return (
        result,
        json.dumps(factual, ensure_ascii=False),
        len(executions),
        repair_count,
    )


def make_executor_node(
    model: RoleModel, runner: LocalPythonRunner | None = None
) -> Node:
    """Create an Executor that selects and runs one local capability."""
    runner = runner or LocalPythonRunner()

    def executor(state: AgentState) -> dict[str, object]:
        if not state.get("structured_plan", False):
            messages = build_executor_messages(
                question=state["question"],
                input_context=state["input_context"],
                plan=state["plan"],
            )
            execution_result = model.generate(role="executor", messages=messages)
            goal_result = GoalResult(
                goal_id="legacy_goal",
                success=True,
                strategy="generated_python",
                capability_name=None,
                result={"legacy_execution_result": execution_result},
                warnings=[],
                error=None,
                artifact_paths=[],
            )
            return {
                "execution_result": execution_result,
                "current_strategy": {
                    "strategy": "generated_python",
                    "capability_name": None,
                    "arguments": {},
                    "concise_reason": "Legacy V0 scripted execution.",
                },
                "current_goal_result": goal_result.model_dump(mode="json"),
                "trace": _trace(state, "executor"),
            }

        registry = _registry(state)
        catalog = registry.catalog()
        messages = build_executor_messages(
            question=state["question"],
            input_context=state["input_context"],
            current_goal=state["current_goal"],
            completed_goal_results=list(state.get("completed_goal_results", [])),
            verification_feedback=(
                state.get("verification_feedback")
                if state.get("verification_decision") == "REPLAN"
                else None
            ),
            capability_catalog=catalog,
            staged_file_paths=_staged_display_paths(state),
        )
        raw_strategy = model.generate(role="executor", messages=messages)
        try:
            strategy = ExecutionStrategy.model_validate_json(raw_strategy)
        except ValidationError as error:
            raise ExecutorOutputError(
                "Executor did not return a valid ExecutionStrategy"
            ) from error
        normalization_warnings: list[str] = []
        if strategy.strategy == "generated_python" and (
            strategy.capability_name is not None or strategy.arguments
        ):
            strategy = strategy.model_copy(
                update={"capability_name": None, "arguments": {}}
            )
            normalization_warnings.append(
                "Normalized generated_python capability_name to null and "
                "arguments to {}."
            )
        if strategy.strategy == "trusted_tool":
            if not strategy.capability_name:
                raise ExecutorOutputError("trusted_tool requires capability_name")
            if strategy.capability_name not in registry.names:
                raise ExecutorOutputError(
                    f"Unknown trusted capability: {strategy.capability_name}"
                )
            tool_result = registry.execute(strategy.capability_name, strategy.arguments)
            goal = IntermediateGoal.model_validate(state["current_goal"])
            goal_result = GoalResult(
                goal_id=goal.goal_id,
                success=tool_result.success,
                strategy="trusted_tool",
                capability_name=strategy.capability_name,
                result=tool_result.output,
                warnings=tool_result.warnings,
                error=tool_result.error,
                artifact_paths=[],
            )
            execution_result = tool_result.model_dump_json()
            script_increment = 0
            repair_increment = 0
            tool_increment = 1
        else:
            goal_result, execution_result, script_increment, repair_increment = (
                _generated_result(
                    state=state,
                    model=model,
                    strategy=strategy,
                    runner=runner,
                )
            )
            tool_increment = 0
        policy_failure_reason = (
            goal_result.error
            if goal_result.error and goal_result.error.startswith("PythonPolicyError:")
            else None
        )
        return {
            "capability_catalog": catalog,
            "current_strategy": strategy.model_dump(mode="json"),
            "current_goal_result": goal_result.model_dump(mode="json"),
            "execution_result": execution_result,
            "policy_failure_reason": policy_failure_reason,
            "trusted_tool_calls": state.get("trusted_tool_calls", 0) + tool_increment,
            "generated_script_count": state.get("generated_script_count", 0)
            + script_increment,
            "code_repair_count": state.get("code_repair_count", 0) + repair_increment,
            "executor_warnings": [
                *state.get("executor_warnings", []),
                *normalization_warnings,
            ],
            "trace": _trace(state, "executor"),
        }

    return executor


def make_verifier_node(model: RoleModel) -> Node:
    """Create a goal-scoped Verifier with one structured-output repair."""

    def verifier(state: AgentState) -> dict[str, object]:
        structured = state.get("structured_plan", False)
        if structured:
            plan = HighLevelPlan.model_validate(state["high_level_plan"])
            current_result = GoalResult.model_validate(state["current_goal_result"])
            messages = build_verifier_messages(
                question=state["question"],
                execution_result=state["execution_result"],
                scientific_objective=plan.scientific_objective,
                current_goal=state["current_goal"],
                strategy=state.get("current_strategy", {}),
                warnings=current_result.warnings,
                prior_goal_results=list(state.get("completed_goal_results", [])),
            )
        else:
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
                if (
                    structured
                    and not GoalResult.model_validate(
                        state["current_goal_result"]
                    ).success
                ):
                    output = VerificationOutput(
                        decision="REPLAN",
                        feedback=(
                            "Execution failed and did not produce a result that can "
                            "satisfy the current goal."
                        ),
                    )
                plan = HighLevelPlan.model_validate(state["high_level_plan"])
                index = state.get("current_goal_index", 0)
                has_more = index + 1 < len(plan.goals)
                can_replan = state.get("replan_count", 0) < state.get("max_replans", 1)
                if output.decision == "PASS":
                    route = (
                        "Verifier -> Select Current Goal"
                        if has_more
                        else "Verifier -> Final Answer Generator"
                    )
                else:
                    route = (
                        "Verifier -> Planner"
                        if can_replan
                        else "Verifier -> Failure Finalizer"
                    )
                record: dict[str, object] = {
                    "iteration": len(state.get("iteration_history", [])) + 1,
                    "plan": state["plan"],
                    "execution_result": state["execution_result"],
                    "verification_decision": output.decision,
                    "verification_feedback": output.feedback,
                    "route": route,
                }
                if structured:
                    record.update(
                        {
                            "goal_id": state["current_goal"]["goal_id"],
                            "strategy": state["current_strategy"]["strategy"],
                            "capability_name": state["current_strategy"].get(
                                "capability_name"
                            ),
                        }
                    )
                updates: dict[str, object] = {
                    "verification_decision": output.decision,
                    "verification_feedback": output.feedback,
                    "trace": _trace(state, f"verifier:{output.decision}"),
                    "iteration_history": [
                        *state.get("iteration_history", []),
                        record,
                    ],
                }
                if output.decision == "PASS":
                    updates["completed_goal_results"] = [
                        *state.get("completed_goal_results", []),
                        state["current_goal_result"],
                    ]
                    updates["current_goal_index"] = index + 1
                return updates
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
    """Generate candidate JSON using only Verifier-approved GoalResults."""

    def final_answer_generator(state: AgentState) -> dict[str, object]:
        request = FinalGenerationRequest(
            question=state["question"],
            approved_execution_result=state["execution_result"],
            verifier_decision=state["verification_decision"],
            verifier_feedback=state["verification_feedback"],
            iteration_history=list(state.get("iteration_history", [])),
            completed_goal_results=list(state.get("completed_goal_results", [])),
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
            validated_data, ensure_ascii=False, indent=2, allow_nan=False
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
    record: OutputValidationRecord = {
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
    """Create one formatting-only repair with an intentionally narrow input."""

    def output_repair(state: AgentState) -> dict[str, object]:
        approved_result = state["execution_result"]
        if state.get("structured_plan", False):
            approved_result = json.dumps(
                {"completed_goal_results": state.get("completed_goal_results", [])},
                ensure_ascii=False,
            )
        request = OutputRepairRequest(
            invalid_raw_output=state["raw_final_output"],
            validation_error=state["output_validation_error"],
            required_schema=FinalAnswer.model_json_schema(),
            approved_execution_result=approved_result,
        )
        return {
            "raw_repair_output": provider.repair(request),
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


def python_policy_failure_node(state: AgentState) -> dict[str, object]:
    """Stop policy-blocked code before it can consume scientific replan budget."""
    reason = state.get("policy_failure_reason", "PythonPolicyError")
    failure = FinalFailureAnswer(
        status="python_policy_failure",
        answer=None,
        key_results={},
        limitations=[
            "Generated code was blocked by the local file-access policy before "
            "execution."
        ],
        error=str(reason),
    )
    return {
        "final_answer": failure.model_dump_json(indent=2),
        "validated_final_answer": failure.model_dump(mode="json"),
        "status": "python_policy_failure",
        "final_status": "python_policy_failure",
        "trace": _trace(state, "failure_finalizer:python_policy"),
    }


def output_failure_node(state: AgentState) -> dict[str, object]:
    """Terminate after formatting repair without claiming completion."""
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
