"""Node factories for the bounded goal-driven verification-first graph."""

from __future__ import annotations

import ast
import hashlib
import json
import mimetypes
import re
import uuid
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from data_analysis_agent.config import code_repair_settings, max_planner_repair_attempts
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
    build_planner_repair_messages,
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
    GoalArtifact,
    GoalArtifactDeclaration,
    GoalResult,
    HighLevelPlan,
    IntermediateGoal,
    PythonGeneration,
    PythonRepair,
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


class GoalArtifactDeclarationError(ValueError):
    """Raised when a script tries to publish an unsafe or missing artifact."""


def partial_run_finalizer_node(state: AgentState) -> dict[str, object]:
    """Finish an intentional goal-limited smoke run without making an answer."""
    completed = list(state.get("completed_goal_results", []))
    target = state.get("stop_after_goals")
    if not target or len(completed) < target:
        raise RuntimeError("partial-run finalizer reached before its requested target")
    return {
        "status": "partial_smoke_completed",
        "final_status": "partial_smoke_completed",
        "partial_run_reached": True,
        "trace": _trace(state, "partial_run:target_reached"),
    }


def _trace(state: AgentState, event: str) -> list[str]:
    return [*state.get("trace", []), event]


def _failure_fingerprint(final: PythonExecutionResult) -> str | None:
    """Normalize a concrete failure while dropping run-specific traceback noise."""
    category = final.failure_category
    if category is None:
        return None
    detail = (
        final.stderr
        if category in {"runtime_error", "syntax_error"} and final.stderr
        else final.error or final.stderr or "unknown failure"
    )
    lines = [line.strip() for line in detail.splitlines() if line.strip()]
    exception_lines = [
        line
        for line in lines
        if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):", line)
    ]
    meaningful = exception_lines[-1] if exception_lines else lines[-1]
    meaningful = re.sub(r"/[^\s'\"]+", "<path>", meaningful)
    meaningful = re.sub(r"\bline \d+\b", "line <n>", meaningful)
    meaningful = re.sub(r"\bv\d+\.py\b", "v<n>.py", meaningful)
    meaningful = re.sub(r"0x[0-9a-fA-F]+", "<address>", meaningful)
    meaningful = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<uuid>", meaningful)
    return f"{category}|{meaningful}"


def _validate_plan(raw_output: str) -> tuple[HighLevelPlan, bool]:
    try:
        plan = HighLevelPlan.model_validate_json(raw_output)
    except ValidationError as error:
        if not raw_output.lstrip().startswith("1."):
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


def _goal_fingerprint(goal: IntermediateGoal | dict[str, object]) -> str:
    """Return a stable identity for a goal definition, not merely its title."""
    payload = (
        goal.model_dump(mode="json")
        if isinstance(goal, IntermediateGoal)
        else IntermediateGoal.model_validate(goal).model_dump(mode="json")
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _completed_goal_fingerprints(state: AgentState) -> dict[str, str]:
    previous = HighLevelPlan.model_validate(state["high_level_plan"])
    completed_ids = {
        str(item.get("goal_id")) for item in state.get("completed_goal_results", [])
    }
    return {
        goal.goal_id: _goal_fingerprint(goal)
        for goal in previous.goals
        if goal.goal_id in completed_ids
    }


def _validate_replan_continuity(state: AgentState, revised_plan: HighLevelPlan) -> None:
    """Reject silent loss or mutation of already verified scientific work."""
    previous = HighLevelPlan.model_validate(state["high_level_plan"])
    previous_by_id = {goal.goal_id: goal for goal in previous.goals}
    revised_by_id = {goal.goal_id: goal for goal in revised_plan.goals}
    for goal_id, fingerprint in _completed_goal_fingerprints(state).items():
        revised_goal = revised_by_id.get(goal_id)
        if revised_goal is None:
            raise PlannerOutputError(
                f"scientific replan omitted completed goal {goal_id}"
            )
        if _goal_fingerprint(revised_goal) != fingerprint:
            previous_goal = previous_by_id[goal_id]
            if previous_goal.goal_id == revised_goal.goal_id:
                raise PlannerOutputError(
                    "completed goal ID "
                    f"{goal_id} was reused with a different goal definition"
                )
            raise PlannerOutputError(f"completed goal {goal_id} was mutated")


def _preserve_completed_results(
    *,
    state: AgentState,
    revised_plan: HighLevelPlan,
) -> list[dict[str, object]]:
    previous = HighLevelPlan.model_validate(state["high_level_plan"])
    previous_goals = {goal.goal_id: _goal_fingerprint(goal) for goal in previous.goals}
    revised_goals = {
        goal.goal_id: _goal_fingerprint(goal) for goal in revised_plan.goals
    }
    return [
        dict(result)
        for result in state.get("completed_goal_results", [])
        if (
            str(result.get("goal_id")) in revised_goals
            and previous_goals.get(str(result.get("goal_id")))
            == revised_goals[str(result.get("goal_id"))]
        )
    ]


def _preserve_approved_artifacts(
    *, state: AgentState, completed: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Archive artifacts from invalidated goals while retaining their files on disk."""
    retained_goal_ids = {str(item.get("goal_id")) for item in completed}
    return [
        artifact.model_dump(mode="json")
        for artifact in _active_approved_artifacts(state)
        if artifact.producer_goal_id in retained_goal_ids
    ]


def _planner_response_update(
    *,
    state: AgentState,
    raw_response: str,
    planner_mode: str,
    planner_repair_count: int,
    run_directory: str,
) -> dict[str, object]:
    """Persist raw Planner output before deterministic validation can reject it."""
    history = list(state.get("planner_response_history", []))
    version = len(history) + 1
    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / f"planner_response_v{version}.json"
    raw_path.write_text(raw_response, encoding="utf-8")
    history.append(
        {
            "mode": planner_mode,
            "version": version,
            "raw_response": raw_response,
            "raw_response_path": str(raw_path),
            "planner_repair_count": planner_repair_count,
            "scientific_replan_count": state.get("replan_count", 0),
        }
    )
    return {
        "plan": raw_response,
        "planner_raw_response": raw_response,
        "planner_raw_response_path": str(raw_path),
        "planner_response_history": history,
        "planner_mode": planner_mode,
        "planner_repair_count": planner_repair_count,
    }


def make_planner_node(model: RoleModel) -> Node:
    """Create a Planner that persists raw output before structural validation."""

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
            completed_goal_fingerprints=(
                _completed_goal_fingerprints(state) if is_replan else None
            ),
            approved_artifacts=(
                [
                    artifact.model_dump(mode="json")
                    for artifact in _active_approved_artifacts(state)
                ]
                if is_replan
                else None
            ),
            current_goal_failure=(
                state.get("current_goal_result") if is_replan else None
            ),
        )
        run_id = state.get("run_id") or uuid.uuid4().hex[:12]
        run_directory = state.get("run_directory") or str(
            Path.cwd() / "runs" / f"run_{run_id}"
        )
        raw_plan = model.generate(role="planner", messages=messages)
        mode = "scientific_replan" if is_replan else "initial"
        return {
            "replan_count": replan_count,
            "max_replans": state.get("max_replans", 1),
            "run_id": run_id,
            "run_directory": run_directory,
            "trusted_tool_calls": state.get("trusted_tool_calls", 0),
            "generated_script_count": state.get("generated_script_count", 0),
            "code_repair_count": state.get("code_repair_count", 0),
            "max_planner_repairs": state.get(
                "max_planner_repairs", max_planner_repair_attempts()
            ),
            "planner_validation_error": None,
            "planner_validation_history": list(
                state.get("planner_validation_history", [])
            ),
            **_planner_response_update(
                state=state,
                raw_response=raw_plan,
                planner_mode=mode,
                planner_repair_count=0,
                run_directory=run_directory,
            ),
            "trace": _trace(state, f"planner:{mode}"),
        }

    return planner


def planner_validator_node(state: AgentState) -> dict[str, object]:
    """Deterministically validate persisted Planner output and expose routing facts."""
    raw_response = state["planner_raw_response"]
    latest = state["planner_response_history"][-1]
    version = int(latest["version"])
    validation_path = (
        Path(state["run_directory"]) / f"planner_validation_v{version}.json"
    )
    mode = state.get("planner_mode", "initial")
    try:
        plan, structured = _validate_plan(raw_response)
        if state.get("planner_mode") == "scientific_replan" and structured:
            _validate_replan_continuity(state, plan)
    except PlannerOutputError as error:
        validation_error = f"{type(error).__name__}: {error}"
        route = (
            "planner_repair"
            if state.get("planner_repair_count", 0)
            < state.get("max_planner_repairs", max_planner_repair_attempts())
            else "planner_output_failure"
        )
        validation = {
            "valid": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "attempt": version,
        }
        validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        record = {
            "mode": mode,
            "version": version,
            "raw_response_path": state["planner_raw_response_path"],
            "validation_path": str(validation_path),
            "valid": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "planner_repair_count": state.get("planner_repair_count", 0),
            "scientific_replan_count": state.get("replan_count", 0),
            "route": route,
        }
        return {
            "planner_validation_error": validation_error,
            "planner_validation_history": [
                *state.get("planner_validation_history", []),
                record,
            ],
            "trace": _trace(state, "planner_validation:INVALID"),
        }
    validation_path.write_text(
        json.dumps(
            {"valid": True, "error_type": None, "error": None, "attempt": version},
            indent=2,
        ),
        encoding="utf-8",
    )
    is_replan = mode == "scientific_replan"
    completed = (
        _preserve_completed_results(state=state, revised_plan=plan)
        if is_replan and structured
        else ([] if is_replan else list(state.get("completed_goal_results", [])))
    )
    approved_artifacts = (
        _preserve_approved_artifacts(state=state, completed=completed)
        if is_replan and structured
        else ([] if is_replan else list(state.get("approved_goal_artifacts", [])))
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
    record = {
        "mode": mode,
        "version": version,
        "raw_response_path": state["planner_raw_response_path"],
        "validation_path": str(validation_path),
        "valid": True,
        "error_type": None,
        "error": None,
        "planner_repair_count": state.get("planner_repair_count", 0),
        "scientific_replan_count": state.get("replan_count", 0),
        "route": "select_current_goal",
    }
    return {
        "high_level_plan": plan.model_dump(mode="json"),
        "structured_plan": structured,
        "completed_goal_results": completed,
        "approved_goal_artifacts": approved_artifacts,
        "current_goal_index": next_index,
        "plan_revision": state.get("plan_revision", 0) + int(is_replan),
        "is_scientific_replan": is_replan,
        "total_goal_count": len(plan.goals),
        "preserved_completed_count": len(completed_ids),
        "remaining_goal_count": len(plan.goals) - len(completed_ids),
        "invalidated_goal_ids": [],
        "new_goal_ids": [
            goal.goal_id
            for goal in plan.goals
            if goal.goal_id
            not in {
                item.get("goal_id")
                for item in state.get("high_level_plan", {}).get("goals", [])
                if isinstance(item, dict)
            }
        ]
        if is_replan
        else [],
        "planner_validation_error": None,
        "planner_validation_history": [
            *state.get("planner_validation_history", []),
            record,
        ],
        "trace": _trace(state, "planner_validation:VALID"),
    }


def make_planner_repair_node(model: RoleModel) -> Node:
    """Repair only Planner JSON/schema structure without scientific replanning."""

    def planner_repair(state: AgentState) -> dict[str, object]:
        attempt = state.get("planner_repair_count", 0) + 1
        raw_response = model.generate(
            role="planner",
            messages=build_planner_repair_messages(
                question=state["question"],
                invalid_response=state["planner_raw_response"],
                validation_error=state.get("planner_validation_error", "Unknown error"),
                previous_plan=(
                    state.get("high_level_plan")
                    if state.get("planner_mode") == "scientific_replan"
                    else None
                ),
                completed_goal_fingerprints=(
                    _completed_goal_fingerprints(state)
                    if state.get("planner_mode") == "scientific_replan"
                    else None
                ),
                approved_artifacts=(
                    [
                        artifact.model_dump(mode="json")
                        for artifact in _active_approved_artifacts(state)
                    ]
                    if state.get("planner_mode") == "scientific_replan"
                    else None
                ),
            ),
        )
        return {
            **_planner_response_update(
                state=state,
                raw_response=raw_response,
                planner_mode=state.get("planner_mode", "initial"),
                planner_repair_count=attempt,
                run_directory=state["run_directory"],
            ),
            "trace": _trace(state, f"planner_repair:attempt_{attempt}"),
        }

    return planner_repair


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
    return {
        "current_goal": goal.model_dump(mode="json"),
        # A new goal (including a replanned one) starts a fresh mechanical budget.
        "code_repair_attempts_for_current_goal": 0,
        "code_repair_no_progress_count": 0,
        "code_repair_no_progress": False,
        "consecutive_failure_family": None,
        "consecutive_failure_fingerprint": None,
        "generated_execution_history": [],
        "python_response_history": [],
        "current_generated_code": "",
        "execution_failure_category": None,
        "pending_goal_artifacts": [],
    }


def _staged_paths(state: AgentState) -> list[Path]:
    supplied = state.get("staged_file_paths", state.get("file_paths", []))
    return [Path(path).resolve() for path in supplied if Path(path).is_file()]


def _staged_display_paths(state: AgentState) -> list[str]:
    supplied = state.get("staged_file_display_paths")
    if supplied is not None:
        return list(supplied)
    return [str(path) for path in _staged_paths(state)]


def _active_approved_artifacts(state: AgentState) -> list[GoalArtifact]:
    """Load the authoritative verifier-approved registry from graph state."""
    return [
        GoalArtifact.model_validate(item)
        for item in state.get("approved_goal_artifacts", [])
    ]


def _artifacts_available_to_current_goal(state: AgentState) -> list[GoalArtifact]:
    """Expose only outputs of this goal's completed prerequisite goals."""
    current = state.get("current_goal")
    if not isinstance(current, dict):
        return []
    dependencies = {str(goal_id) for goal_id in current.get("depends_on", [])}
    completed = {
        str(item.get("goal_id")) for item in state.get("completed_goal_results", [])
    }
    return [
        artifact
        for artifact in _active_approved_artifacts(state)
        if artifact.producer_goal_id in dependencies
        and artifact.producer_goal_id in completed
        and Path(artifact.path).is_file()
    ]


def _allowed_paths(state: AgentState) -> list[Path]:
    """Combine public staged inputs with dependency-safe approved artifacts."""
    paths = [*_staged_paths(state)]
    paths.extend(
        Path(item.path).resolve()
        for item in _artifacts_available_to_current_goal(state)
    )
    return list(dict.fromkeys(paths))


def _approved_artifact_context(state: AgentState) -> list[dict[str, str | None]]:
    """Provide model-facing provenance without exposing arbitrary goal files."""
    return [
        {
            "producer_goal_id": artifact.producer_goal_id,
            "path": artifact.path,
            "description": artifact.description,
            "media_type": artifact.media_type,
        }
        for artifact in _artifacts_available_to_current_goal(state)
    ]


def _is_private_or_diagnostic_artifact(relative_name: Path) -> bool:
    """Reject runner, source, model, and workflow diagnostics by stable names."""
    if "generated_outputs" in relative_name.parts:
        return True
    name = relative_name.name
    reserved_prefixes = (
        "generated_code_v",
        "runner_entry_v",
        "stdout",
        "stderr",
        "execution_result",
        "artifact_metadata",
        "python_generation",
        "python_repair",
        "planner_response",
        "planner_validation",
    )
    return name == "workflow.log" or name.startswith(reserved_prefixes)


def _declared_goal_artifacts(
    *, goal: IntermediateGoal, goal_directory: Path, result: dict[str, object]
) -> list[GoalArtifact]:
    """Validate explicit, goal-local analysis outputs before verifier approval."""
    raw_declarations = result.get("artifacts", [])
    if not isinstance(raw_declarations, list):
        raise GoalArtifactDeclarationError("artifacts must be a list when supplied")
    try:
        declarations = [
            GoalArtifactDeclaration.model_validate(item) for item in raw_declarations
        ]
    except ValidationError as error:
        raise GoalArtifactDeclarationError(
            "artifacts must contain only typed artifact declarations"
        ) from error
    root = goal_directory.resolve()
    artifacts: list[GoalArtifact] = []
    seen_names: set[str] = set()
    for declaration in declarations:
        relative_name = Path(declaration.relative_name)
        if (
            relative_name.is_absolute()
            or not relative_name.parts
            or ".." in relative_name.parts
            or _is_private_or_diagnostic_artifact(relative_name)
        ):
            raise GoalArtifactDeclarationError(
                "artifact path is not an eligible analysis output: "
                f"{declaration.relative_name}"
            )
        if declaration.relative_name in seen_names:
            raise GoalArtifactDeclarationError(
                f"artifact was declared more than once: {declaration.relative_name}"
            )
        seen_names.add(declaration.relative_name)
        path = (root / relative_name).resolve()
        if root not in path.parents or not path.is_file():
            raise GoalArtifactDeclarationError(
                f"declared artifact does not exist inside the current goal directory: "
                f"{declaration.relative_name}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts.append(
            GoalArtifact(
                artifact_id=f"{goal.goal_id}:{declaration.relative_name}:{digest[:12]}",
                producer_goal_id=goal.goal_id,
                path=str(path),
                relative_name=declaration.relative_name,
                media_type=(
                    declaration.media_type
                    if declaration.media_type is not None
                    else mimetypes.guess_type(path.name)[0]
                ),
                description=declaration.description,
                size_bytes=path.stat().st_size,
                sha256=digest,
            )
        )
    return artifacts


def _registry(state: AgentState) -> TrustedToolRegistry:
    allowed = _allowed_paths(state)
    roots = list(dict.fromkeys(path.parent for path in allowed)) or [Path.cwd()]
    return TrustedToolRegistry(allowed_roots=roots, allowed_files=allowed)


class _CanonicalNames(ast.NodeTransformer):
    """Normalize local identifier spelling for material-change diagnostics."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id == "__agent_result__":
            return node
        normalized = self.names.setdefault(node.id, f"name_{len(self.names)}")
        return ast.copy_location(ast.Name(id=normalized, ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:  # noqa: N802
        normalized = self.names.setdefault(node.arg, f"name_{len(self.names)}")
        return ast.copy_location(
            ast.arg(arg=normalized, annotation=node.annotation), node
        )


def _canonical_source(code: str) -> str:
    """Ignore comments, formatting, and local variable spelling when possible."""
    try:
        tree = _CanonicalNames().visit(ast.parse(code))
        ast.fix_missing_locations(tree)
        return ast.dump(tree, include_attributes=False)
    except SyntaxError:
        return "".join(line.split("#", 1)[0].strip() for line in code.splitlines())


def _materially_changed(previous: str, current: str) -> bool:
    return _canonical_source(previous) != _canonical_source(current)


def _next_python_version(state: AgentState) -> int:
    return len(state.get("python_response_history", [])) + 1


def _python_contract_response(
    *,
    model: RoleModel,
    state: AgentState,
    messages: list[dict[str, str]],
    contract_class: type[PythonGeneration] | type[PythonRepair],
    artifact_prefix: str,
) -> tuple[
    PythonGeneration | PythonRepair | None,
    int,
    list[dict[str, object]],
    list[str],
    str | None,
]:
    """Persist and validate a strict response before any source reaches the runner."""
    version = _next_python_version(state)
    goal = IntermediateGoal.model_validate(state["current_goal"])
    goal_directory = Path(state["run_directory"]) / "goals" / goal.goal_id
    goal_directory.mkdir(parents=True, exist_ok=True)
    raw = model.generate_structured(
        role="executor",
        messages=messages,
        schema_name=artifact_prefix,
        schema=contract_class.model_json_schema(),
    )
    try:
        contract = contract_class.model_validate_json(raw)
    except ValidationError as error:
        raw_path = goal_directory / f"{artifact_prefix}_invalid_v{version}.txt"
        validation_path = (
            goal_directory / f"{artifact_prefix}_validation_v{version}.json"
        )
        raw_path.write_text(raw, encoding="utf-8")
        validation_error = f"{type(error).__name__}: {error}"
        validation_path.write_text(
            json.dumps(
                {
                    "valid": False,
                    "failure_category": "generation_contract_error",
                    "error": validation_error,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        history = [
            *state.get("python_response_history", []),
            {
                "version": version,
                "contract": artifact_prefix,
                "valid": False,
                "raw_response_path": str(raw_path),
                "validation_path": str(validation_path),
                "error": validation_error,
            },
        ]
        return (
            None,
            version,
            history,
            [str(raw_path), str(validation_path)],
            validation_error,
        )
    metadata_path = goal_directory / f"{artifact_prefix}_v{version}.json"
    metadata_path.write_text(contract.model_dump_json(indent=2), encoding="utf-8")
    history = [
        *state.get("python_response_history", []),
        {
            "version": version,
            "contract": artifact_prefix,
            "valid": True,
            "metadata_path": str(metadata_path),
            "error": None,
        },
    ]
    return contract, version, history, [str(metadata_path)], None


def _generated_execution_update(
    *,
    state: AgentState,
    strategy: ExecutionStrategy,
    runner: LocalPythonRunner,
    code: str,
    materially_changed: bool,
    repair_attempts: int,
    version: int,
    response_artifacts: list[str],
) -> dict[str, object]:
    """Run one version and expose only typed execution facts to graph routing."""
    goal = IntermediateGoal.model_validate(state["current_goal"])
    goal_directory = Path(state["run_directory"]) / "goals" / goal.goal_id
    final = runner.run(
        code=code,
        goal_directory=goal_directory,
        allowed_files=_allowed_paths(state),
        version=version,
    )
    pending_artifacts: list[GoalArtifact] = []
    if final.success:
        try:
            pending_artifacts = _declared_goal_artifacts(
                goal=goal,
                goal_directory=goal_directory,
                result=dict(final.result),
            )
        except GoalArtifactDeclarationError as error:
            final.success = False
            final.error = f"ResultContractError: {error}"
            final.failure_category = "result_contract_error"
    final.artifact_paths = list(
        dict.fromkeys([*final.artifact_paths, *response_artifacts])
    )
    updates = _finalize_generated_attempt(
        state=state,
        strategy=strategy,
        final=final,
        code=code,
        materially_changed=materially_changed,
        repair_attempts=repair_attempts,
        generated_script_increment=1,
    )
    updates["pending_goal_artifacts"] = [
        artifact.model_dump(mode="json") for artifact in pending_artifacts
    ]
    return updates


def _generation_contract_failure_update(
    *,
    state: AgentState,
    strategy: ExecutionStrategy,
    version: int,
    error: str,
    response_artifacts: list[str],
    repair_attempts: int,
) -> dict[str, object]:
    """Represent invalid structured output without ever creating a Python file."""
    goal = IntermediateGoal.model_validate(state["current_goal"])
    goal_directory = Path(state["run_directory"]) / "goals" / goal.goal_id
    final = PythonExecutionResult(
        success=False,
        version=version,
        exit_code=None,
        stdout="",
        stderr="",
        result={},
        error=f"GenerationContractError: {error}",
        duration_seconds=0.0,
        script_path="",
        artifact_paths=list(response_artifacts),
        policy_validated=False,
        parsed_result=False,
        failure_category="generation_contract_error",
        deterministic_result_recovery_attempted=False,
    )
    execution_path = goal_directory / f"execution_result_v{version}.json"
    execution_path.write_text(final.model_dump_json(indent=2), encoding="utf-8")
    final.artifact_paths.append(str(execution_path))
    (goal_directory / "execution_result.json").write_text(
        final.model_dump_json(indent=2), encoding="utf-8"
    )
    return _finalize_generated_attempt(
        state=state,
        strategy=strategy,
        final=final,
        code=state.get("current_generated_code", ""),
        materially_changed=False,
        repair_attempts=repair_attempts,
        generated_script_increment=0,
    )


def _finalize_generated_attempt(
    *,
    state: AgentState,
    strategy: ExecutionStrategy,
    final: PythonExecutionResult,
    code: str,
    materially_changed: bool,
    repair_attempts: int,
    generated_script_increment: int,
) -> dict[str, object]:
    """Project one execution or generation-contract failure into AgentState."""
    goal = IntermediateGoal.model_validate(state["current_goal"])
    goal_directory = Path(state["run_directory"]) / "goals" / goal.goal_id
    executions = [
        PythonExecutionResult.model_validate(item)
        for item in state.get("generated_execution_history", [])
    ] + [final]
    metadata_path = write_execution_metadata(
        goal_directory=goal_directory,
        run_id=state["run_id"],
        goal_id=goal.goal_id,
        executions=executions,
        strategy_reason=strategy.concise_reason,
    )
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
        "repair_required": repair_attempts > 0,
        "timed_out": final.timed_out,
        "policy_validated": final.policy_validated,
        "parsed_result": final.parsed_result,
        "failure_category": final.failure_category,
        "artifact_count": saved_artifact_count,
        "artifact_directory": str(goal_directory),
        "latest_stderr_path": str(latest_stderr_path),
        "artifact_paths": artifact_paths,
    }
    max_repairs, max_no_progress = code_repair_settings()
    repair_limit = state.get("max_code_repair_attempts", max_repairs)
    no_progress_limit = state.get(
        "max_code_repair_no_progress_attempts", max_no_progress
    )
    failure_family = final.failure_category
    failure_fingerprint = _failure_fingerprint(final)
    if final.success:
        consecutive_family_count = 0
        failure_family = None
        failure_fingerprint = None
    elif failure_fingerprint == state.get("consecutive_failure_fingerprint"):
        consecutive_family_count = state.get("code_repair_no_progress_count", 0) + 1
    else:
        consecutive_family_count = 1
    no_progress = not final.success and consecutive_family_count >= no_progress_limit
    factual.update(
        {
            "normalized_failure_family": failure_family,
            "normalized_failure_fingerprint": failure_fingerprint,
            "consecutive_failure_family_count": consecutive_family_count,
            "materially_changed": materially_changed,
            "deterministic_result_recovery_attempted": (
                final.deterministic_result_recovery_attempted
            ),
        }
    )
    if final.success:
        next_route = "verifier"
    elif no_progress:
        next_route = "mechanical_failure"
    elif repair_attempts < repair_limit:
        next_route = "mechanical_repair"
    else:
        next_route = "mechanical_failure"
    record = {
        "goal_id": goal.goal_id,
        "version": final.version,
        "attempt": repair_attempts,
        "failure_category": final.failure_category,
        "normalized_failure_family": failure_family,
        "normalized_failure_fingerprint": failure_fingerprint,
        "consecutive_failure_family_count": consecutive_family_count,
        "exit_code": final.exit_code,
        "timed_out": final.timed_out,
        "policy_validated": final.policy_validated,
        "parsed_result": final.parsed_result,
        "error": final.error,
        "source_changed": materially_changed,
        "materially_changed": materially_changed,
        "deterministic_result_recovery_attempted": (
            final.deterministic_result_recovery_attempted
        ),
        "route": next_route,
        "code_repair_attempts_for_current_goal": repair_attempts,
        "scientific_replan_count": state.get("replan_count", 0),
    }
    trace_event = (
        "code_execution:success"
        if final.success
        else f"code_execution:{final.failure_category}"
    )
    return {
        "current_goal_result": result.model_dump(mode="json"),
        "execution_result": json.dumps(factual, ensure_ascii=False),
        "current_generated_code": code,
        "generated_execution_history": [
            item.model_dump(mode="json") for item in executions
        ],
        "python_response_history": list(state.get("python_response_history", [])),
        "code_execution_history": [*state.get("code_execution_history", []), record],
        "execution_failure_category": final.failure_category,
        "failure_category": final.failure_category,
        "policy_failure_reason": (
            final.error if final.failure_category == "policy_error" else None
        ),
        "generated_script_count": state.get("generated_script_count", 0)
        + generated_script_increment,
        "code_repair_attempts_for_current_goal": repair_attempts,
        "max_code_repair_attempts": repair_limit,
        "code_repair_no_progress_count": consecutive_family_count,
        "consecutive_failure_family": failure_family,
        "consecutive_failure_fingerprint": failure_fingerprint,
        "max_code_repair_no_progress_attempts": no_progress_limit,
        "code_repair_no_progress": no_progress,
        "trace": _trace(state, trace_event),
        "pending_goal_artifacts": [],
    }


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
            staged_file_paths=[str(path) for path in _allowed_paths(state)],
            approved_artifacts=_approved_artifact_context(state),
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
            goal = IntermediateGoal.model_validate(state["current_goal"])
            goal_directory = Path(state["run_directory"]) / "goals" / goal.goal_id
            source_messages = build_python_generation_messages(
                current_goal=goal.model_dump(mode="json"),
                staged_file_paths=[str(path) for path in _allowed_paths(state)],
                completed_goal_results=list(state.get("completed_goal_results", [])),
                goal_directory=str(goal_directory),
                approved_artifacts=_approved_artifact_context(state),
            )
            contract, version, response_history, artifacts, contract_error = (
                _python_contract_response(
                    model=model,
                    state=state,
                    messages=source_messages,
                    contract_class=PythonGeneration,
                    artifact_prefix="python_generation",
                )
            )
            generation_state: AgentState = {
                **state,
                "python_response_history": response_history,
                "trace": _trace(state, "executor"),
            }
            if contract is None:
                generated = _generation_contract_failure_update(
                    state=generation_state,
                    strategy=strategy,
                    version=version,
                    error=contract_error or "invalid PythonGeneration response",
                    response_artifacts=artifacts,
                    repair_attempts=0,
                )
            else:
                if not isinstance(contract, PythonGeneration):
                    raise AssertionError("unexpected Python generation contract")
                generated = _generated_execution_update(
                    state=generation_state,
                    strategy=strategy,
                    runner=runner,
                    code=contract.source(),
                    materially_changed=True,
                    repair_attempts=0,
                    version=version,
                    response_artifacts=artifacts,
                )
            return {
                "capability_catalog": catalog,
                "current_strategy": strategy.model_dump(mode="json"),
                "trusted_tool_calls": state.get("trusted_tool_calls", 0),
                "code_repair_count": state.get("code_repair_count", 0),
                "executor_warnings": [
                    *state.get("executor_warnings", []),
                    *normalization_warnings,
                ],
                **generated,
            }

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


def make_mechanical_repair_node(
    model: RoleModel, runner: LocalPythonRunner | None = None
) -> Node:
    """Repair a failed implementation without changing the approved goal."""
    runner = runner or LocalPythonRunner()

    def mechanical_repair(state: AgentState) -> dict[str, object]:
        goal = IntermediateGoal.model_validate(state["current_goal"])
        previous = PythonExecutionResult.model_validate(
            state["generated_execution_history"][-1]
        )
        attempt = state.get("code_repair_attempts_for_current_goal", 0) + 1
        history = state.get("code_execution_history", [])[-3:]
        messages = build_python_repair_messages(
            current_goal=goal.model_dump(mode="json"),
            code=state["current_generated_code"],
            failure_category=previous.failure_category or "runtime_error",
            stdout=previous.stdout,
            stderr=previous.stderr,
            error=previous.error,
            failure_fingerprint=state.get("consecutive_failure_fingerprint"),
            staged_file_paths=[str(path) for path in _allowed_paths(state)],
            goal_directory=str(Path(state["run_directory"]) / "goals" / goal.goal_id),
            repair_history=history,
            approved_artifacts=_approved_artifact_context(state),
        )
        contract, version, response_history, artifacts, contract_error = (
            _python_contract_response(
                model=model,
                state=state,
                messages=messages,
                contract_class=PythonRepair,
                artifact_prefix="python_repair",
            )
        )
        repair_state: AgentState = {
            **state,
            "python_response_history": response_history,
            "trace": _trace(state, f"mechanical_repair:attempt_{attempt}"),
        }
        strategy = ExecutionStrategy.model_validate(state["current_strategy"])
        if contract is None:
            updates = _generation_contract_failure_update(
                state=repair_state,
                strategy=strategy,
                version=version,
                error=contract_error or "invalid PythonRepair response",
                response_artifacts=artifacts,
                repair_attempts=attempt,
            )
        else:
            if not isinstance(contract, PythonRepair):
                raise AssertionError("unexpected Python repair contract")
            updates = _generated_execution_update(
                state=repair_state,
                strategy=strategy,
                runner=runner,
                code=contract.source(),
                materially_changed=_materially_changed(
                    state.get("current_generated_code", ""), contract.source()
                ),
                repair_attempts=attempt,
                version=version,
                response_artifacts=artifacts,
            )
        return {
            **updates,
            "code_repair_count": state.get("code_repair_count", 0) + 1,
        }

    return mechanical_repair


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
                    "failure_category": (
                        "scientific_verification_failure"
                        if output.decision == "REPLAN"
                        else None
                    ),
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
                    goal_id = str(state["current_goal"]["goal_id"])
                    pending = [
                        GoalArtifact.model_validate(item)
                        for item in state.get("pending_goal_artifacts", [])
                    ]
                    if any(item.producer_goal_id != goal_id for item in pending):
                        raise GoalArtifactDeclarationError(
                            "pending artifact producer does not match the current goal"
                        )
                    approved = [
                        *[
                            item
                            for item in _active_approved_artifacts(state)
                            if item.producer_goal_id != goal_id
                        ],
                        *pending,
                    ]
                    registry_path = (
                        Path(state["run_directory"]) / "approved_goal_artifacts.json"
                    )
                    registry_path.parent.mkdir(parents=True, exist_ok=True)
                    registry_path.write_text(
                        json.dumps(
                            [item.model_dump(mode="json") for item in approved],
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    updates["approved_goal_artifacts"] = [
                        item.model_dump(mode="json") for item in approved
                    ]
                    updates["approved_goal_artifacts_path"] = str(registry_path)
                    updates["pending_goal_artifacts"] = []
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


def planner_output_failure_node(state: AgentState) -> dict[str, object]:
    """Finalize exhausted Planner structural repair without entering execution."""
    calls = len(state.get("planner_response_history", []))
    repairs = state.get("planner_repair_count", 0)
    error = state.get("planner_validation_error", "Unknown Planner validation error")
    raw_path = state.get("planner_raw_response_path", "not available")
    executor_invoked = bool(state.get("current_strategy"))
    verifier_invoked = bool(state.get("iteration_history"))
    detail = (
        f"planner_calls={calls}; planner_repair_attempts={repairs}; "
        f"final_validation_error={error}; raw_response_path={raw_path}; "
        f"scientific_replan_count={state.get('replan_count', 0)}; "
        f"executor_invoked={str(executor_invoked).lower()}; "
        f"verifier_invoked={str(verifier_invoked).lower()}"
    )
    failure = FinalFailureAnswer(
        status="planner_output_failed",
        answer=None,
        key_results={},
        limitations=[
            "Planner output did not satisfy the deterministic HighLevelPlan "
            "contract after bounded structural repair."
        ],
        error=detail,
    )
    return {
        "final_answer": failure.model_dump_json(indent=2),
        "validated_final_answer": failure.model_dump(mode="json"),
        "status": "planner_output_failed",
        "final_status": "planner_output_failed",
        "trace": _trace(state, "failure_finalizer:planner_output"),
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


def mechanical_execution_failure_node(state: AgentState) -> dict[str, object]:
    """Finalize exhausted generated-code repair without scientific replanning."""
    goal = state.get("current_goal", {}).get("goal_id", "unknown_goal")
    category = state.get("execution_failure_category", "runtime_error")
    error = (
        GoalResult.model_validate(state["current_goal_result"]).error or "Unknown error"
    )
    attempts = state.get("code_repair_attempts_for_current_goal", 0)
    no_progress = state.get("code_repair_no_progress", False)
    reason = "mechanical_repair_no_progress" if no_progress else "code_repair_exhausted"
    detail = (
        f"{reason}; goal_id={goal}; repair_attempts={attempts}; "
        f"final_failure_category={category}; no_progress_termination={no_progress}; "
        f"scientific_replan_count={state.get('replan_count', 0)}; error={error}"
    )
    failure = FinalFailureAnswer(
        status="mechanical_execution_failed",
        answer=None,
        key_results={},
        limitations=[
            "Generated code never produced a valid executor-level result, so it was "
            "not evaluated by the scientific Verifier."
        ],
        error=detail,
    )
    return {
        "final_answer": failure.model_dump_json(indent=2),
        "validated_final_answer": failure.model_dump(mode="json"),
        "status": "mechanical_execution_failed",
        "final_status": "mechanical_execution_failed",
        "trace": _trace(state, f"failure_finalizer:{reason}"),
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
