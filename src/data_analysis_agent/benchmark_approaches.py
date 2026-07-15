"""Independent direct, one-shot-code, and full-agent benchmark approaches."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from data_analysis_agent.benchmark_progress import ProgressCallback, ProgressEvent
from data_analysis_agent.benchmark_types import (
    Approach,
    ApproachOutcome,
    BenchmarkConfig,
    PublicTaskView,
)
from data_analysis_agent.demo import write_workflow_log
from data_analysis_agent.final_output import DeterministicFinalOutputProvider
from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import RecordingRoleModel, RoleModel
from data_analysis_agent.python_runner import LocalPythonRunner
from data_analysis_agent.state import AgentState

ModelFactory = Callable[[Approach, PublicTaskView], RoleModel]


def _progress(progress: ProgressCallback | None, kind: str, **data: object) -> None:
    """Emit presentation facts without terminal formatting in workflow code."""
    if progress is not None:
        progress(ProgressEvent(type=kind, **data))


DIRECT_SYSTEM_PROMPT = """Solve the supplied data-analysis task directly.
Return exactly one JSON object matching the public answer schema. Do not return
Markdown or prose outside the object. This is a single response: no tools,
execution, follow-up, or repair is available."""

ONE_SHOT_CODE_SYSTEM_PROMPT = """Generate one deterministic Python script that
solves the supplied task from the explicitly staged files. Use only the Python
standard library, pandas, numpy, or scipy. Read only the listed files, do not use
network, subprocess, shell, environment-variable, deletion, or package-install
operations, and write only in the execution directory. Print exactly one JSON
object matching the public answer schema as the final non-empty stdout line.
Return only Python source without Markdown fences or explanation. The script is
executed once and will not be repaired."""


def build_direct_answer_messages(public: PublicTaskView) -> list[dict[str, str]]:
    """Include the complete public prompt, schema, filenames, and data contents."""
    sections = []
    for name in public.data_files:
        display_name = Path(name).name
        sections.append(
            f"----- BEGIN FILE: {display_name} -----\n"
            f"{public.data_contents[_content_key(public, name)].rstrip()}\n"
            f"----- END FILE: {display_name} -----"
        )
    user = (
        f"Task ID: {public.task_id}\n\nTask prompt:\n{public.prompt}\n\n"
        "Public answer schema:\n"
        f"{json.dumps(public.answer_schema, ensure_ascii=False)}\n\n"
        "Complete input data:\n" + "\n\n".join(sections)
    )
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_one_shot_code_messages(public: PublicTaskView) -> list[dict[str, str]]:
    """Describe only public staged inputs and the shared prompt/schema."""
    descriptions = [
        {"filename": Path(path).name, "staged_path": path} for path in public.data_files
    ]
    user = (
        f"Task ID: {public.task_id}\n\nTask prompt:\n{public.prompt}\n\n"
        "Public answer schema:\n"
        f"{json.dumps(public.answer_schema, ensure_ascii=False)}\n\n"
        "Staged public data files:\n"
        f"{json.dumps(descriptions, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": ONE_SHOT_CODE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _content_key(public: PublicTaskView, staged_name: str) -> str:
    if staged_name in public.data_contents:
        return staged_name
    basename = Path(staged_name).name
    matches = [name for name in public.data_contents if Path(name).name == basename]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely map staged data file {staged_name}")
    return matches[0]


def _parse_object(raw: str) -> dict[str, JsonValue]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("candidate must be one JSON object")
    return parsed


def _extract_code(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        stripped = "\n".join(lines).strip()
    if not stripped:
        raise ValueError("model returned empty generated Python")
    return stripped + "\n"


def _usage(recorder: RecordingRoleModel) -> tuple[int | None, int | None, int | None]:
    if not recorder.exchanges or any(
        exchange.token_usage is None for exchange in recorder.exchanges
    ):
        return None, None, None
    return (
        sum(exchange.token_usage["prompt_tokens"] for exchange in recorder.exchanges),
        sum(
            exchange.token_usage["completion_tokens"] for exchange in recorder.exchanges
        ),
        sum(exchange.token_usage["total_tokens"] for exchange in recorder.exchanges),
    )


def _call_counts(recorder: RecordingRoleModel) -> tuple[int, int]:
    return (
        sum(exchange.api_request_count for exchange in recorder.exchanges),
        sum(exchange.transport_retry_count for exchange in recorder.exchanges),
    )


def _model_observer(progress: ProgressCallback | None):
    if progress is None:
        return None

    def observe(
        phase: str,
        role: str,
        call_number: int,
        elapsed: float,
        error: str | None,
    ) -> None:
        labels = {
            "planner": "Planner",
            "executor": "Executor",
            "verifier": "Verifier",
            "direct_answer": "Direct answer",
            "one_shot_code": "Code generation",
        }
        label = labels[role]
        if phase == "start":
            _progress(progress, "activity", message=f"{label} — calling model...")
        elif error:
            _progress(
                progress,
                "error",
                error=f"{label} — failed after {elapsed:.1f}s: {error}",
            )
        else:
            _progress(
                progress,
                "activity",
                message=f"{label} — completed in {elapsed:.1f}s",
            )

    return observe


def _infrastructure_error(exception: Exception) -> bool:
    name = type(exception).__name__
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
    } or (name == "InternalServerError")


def _save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolved_public_files(public: PublicTaskView, run_directory: Path) -> list[Path]:
    input_root = (run_directory / "inputs").resolve()
    resolved: list[Path] = []
    for supplied in public.data_files:
        relative = Path(supplied)
        candidate = (run_directory / relative).resolve()
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.parts[0] != "inputs"
            or input_root not in candidate.parents
            or not candidate.is_file()
        ):
            raise ValueError(f"Invalid staged public input path: {supplied}")
        resolved.append(candidate)
    return resolved


def run_direct_answer(
    *,
    public: PublicTaskView,
    model: RoleModel,
    run_directory: Path,
    config: BenchmarkConfig,
    progress: ProgressCallback | None = None,
) -> ApproachOutcome:
    """Make exactly one model call and perform no execution or repair."""
    messages = build_direct_answer_messages(public)
    prompt_chars = sum(len(message["content"]) for message in messages)
    if prompt_chars > config.direct_answer_max_input_chars:
        return ApproachOutcome(
            status="not_applicable",
            not_applicable_reason=(
                f"complete public input is {prompt_chars} characters, above the "
                f"configured {config.direct_answer_max_input_chars}-character limit"
            ),
        )
    recorder = RecordingRoleModel(model, _model_observer(progress))
    not_applicable_reason = None
    exception_class = None
    try:
        raw = recorder.generate(role="direct_answer", messages=messages)
        (run_directory / "raw_response.txt").write_text(raw, encoding="utf-8")
        candidate = _parse_object(raw)
        _save_json(run_directory / "candidate.json", candidate)
        status = "completed"
        error = None
    except (json.JSONDecodeError, ValueError) as exception:
        candidate = None
        status = "invalid_json"
        error = f"{type(exception).__name__}: {exception}"
    except Exception as exception:
        candidate = None
        exception_class = type(exception).__name__
        error = f"{type(exception).__name__}: {exception}"
        lowered = str(exception).lower()
        if "context" in lowered and any(
            word in lowered for word in ("length", "window", "token", "maximum")
        ):
            status = "not_applicable"
            not_applicable_reason = (
                "provider rejected the complete input as context overflow"
            )
        else:
            status = (
                "infrastructure_error" if _infrastructure_error(exception) else "error"
            )
            not_applicable_reason = None
    prompt_tokens, completion_tokens, total_tokens = _usage(recorder)
    api_call_count, retry_count = _call_counts(recorder)
    return ApproachOutcome(
        status=status,
        candidate=candidate,
        api_call_count=api_call_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        transport_retry_count=retry_count,
        run_error=error,
        error_category=("transport_api" if status == "infrastructure_error" else None),
        exception_class=(exception_class if status == "infrastructure_error" else None),
        not_applicable_reason=not_applicable_reason,
    )


def run_one_shot_code(
    *,
    public: PublicTaskView,
    model: RoleModel,
    run_directory: Path,
    config: BenchmarkConfig,
    progress: ProgressCallback | None = None,
) -> ApproachOutcome:
    """Generate once, execute once, and never invoke repair or verification."""
    recorder = RecordingRoleModel(model, _model_observer(progress))
    execution = None
    exception_class = None
    try:
        raw = recorder.generate(
            role="one_shot_code", messages=build_one_shot_code_messages(public)
        )
        (run_directory / "raw_code_response.txt").write_text(raw, encoding="utf-8")
        code = _extract_code(raw)
        execution = LocalPythonRunner(
            timeout_seconds=config.timeout_seconds,
            progress_callback=(
                lambda message: (
                    _progress(progress, "activity", message=message)
                    if progress
                    else None
                )
            ),
        ).run(
            code=code,
            goal_directory=run_directory / "execution",
            allowed_files=_resolved_public_files(public, run_directory),
            version=1,
            working_directory=run_directory,
        )
        if execution.success:
            candidate = execution.result
            _save_json(run_directory / "candidate.json", candidate)
            status = "completed"
            error = None
        else:
            candidate = None
            if execution.timed_out:
                status = "timed_out"
            elif (execution.error or "").startswith("PythonPolicyError:"):
                status = "python_policy_failure"
            elif (execution.error or "").startswith("Invalid JSON output"):
                status = "invalid_json"
            else:
                status = "execution_failed"
            error = execution.error
    except Exception as exception:
        candidate = None
        exception_class = type(exception).__name__
        status = "infrastructure_error" if _infrastructure_error(exception) else "error"
        error = f"{type(exception).__name__}: {exception}"
    prompt_tokens, completion_tokens, total_tokens = _usage(recorder)
    api_call_count, retry_count = _call_counts(recorder)
    return ApproachOutcome(
        status=status,
        candidate=candidate,
        api_call_count=api_call_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        transport_retry_count=retry_count,
        execution_exit_code=execution.exit_code if execution else None,
        timed_out=execution.timed_out if execution else False,
        generated_script_count=1 if execution else 0,
        run_error=error,
        error_category=(
            "transport_api"
            if status == "infrastructure_error"
            else "python_policy"
            if status == "python_policy_failure"
            else None
        ),
        exception_class=(exception_class if status == "infrastructure_error" else None),
    )


def _agent_input_context(public: PublicTaskView) -> str:
    sections = []
    for path in public.data_files:
        key = _content_key(public, path)
        sections.append(
            f"File: {Path(path).name}\n{public.data_contents[key].rstrip()}"
        )
    return "\n\n".join(sections)


def run_agent(
    *,
    public: PublicTaskView,
    model: RoleModel,
    run_directory: Path,
    config: BenchmarkConfig,
    progress: ProgressCallback | None = None,
) -> ApproachOutcome:
    """Invoke the existing full Planner/Executor/Verifier workflow from scratch."""
    recorder = RecordingRoleModel(model, _model_observer(progress))
    agent_directory = run_directory / "agent_run"
    exception_class = None
    try:
        if progress:
            _progress(progress, "workflow_started")
        workflow_started = time.perf_counter()
        staged_files = _resolved_public_files(public, run_directory)
        initial_state: AgentState = {
            "question": public.prompt,
            "file_paths": [Path(path).name for path in public.data_files],
            "staged_file_paths": [str(path) for path in staged_files],
            "staged_file_display_paths": list(public.data_files),
            "execution_working_directory": str(run_directory.resolve()),
            "input_context": _agent_input_context(public),
            "run_directory": str(agent_directory),
            "replan_count": 0,
            "max_replans": 1,
            "output_repair_count": 0,
            "max_output_repairs": 1,
            "output_validation_history": [],
            "trace": [],
            "iteration_history": [],
        }
        workflow = build_graph(
            recorder,
            DeterministicFinalOutputProvider(),
            LocalPythonRunner(
                timeout_seconds=config.timeout_seconds,
                progress_callback=(
                    lambda message: (
                        _progress(progress, "activity", message=message)
                        if progress
                        else None
                    )
                ),
            ),
        )
        if progress is None:
            result = cast(AgentState, workflow.invoke(initial_state))
        else:
            result = initial_state
            trace_size = 0
            current_goal_id: str | None = None
            for snapshot in workflow.stream(initial_state, stream_mode="values"):
                result = cast(AgentState, snapshot)
                goal = result.get("current_goal")
                if goal and goal.get("goal_id") != current_goal_id:
                    current_goal_id = str(goal["goal_id"])
                    _progress(
                        progress,
                        "goal_started",
                        goal_id=current_goal_id,
                        objective=str(goal.get("objective", "")),
                    )
                trace = result.get("trace", [])
                for event in trace[trace_size:]:
                    if event == "executor":
                        strategy = result.get("current_strategy", {}).get(
                            "strategy", "unknown"
                        )
                        _progress(
                            progress,
                            "activity",
                            message=f"Executor — strategy: {strategy}",
                        )
                    elif event == "planner_validation:VALID":
                        plan = result.get("high_level_plan", {})
                        goals = plan.get("goals", []) if isinstance(plan, dict) else []
                        _progress(progress, "plan_available", goals=goals)
                    elif event == "planner_validation:INVALID":
                        _progress(
                            progress, "activity", message="Planner — invalid plan"
                        )
                        _progress(
                            progress,
                            "error",
                            error=str(result.get("planner_validation_error", "")),
                        )
                    elif event.startswith("planner_repair:attempt_"):
                        _progress(
                            progress,
                            "activity",
                            message=(
                                "Planner repair: "
                                f"[{result.get('planner_repair_count', 0)}/"
                                f"{result.get('max_planner_repairs', 2)}]"
                            ),
                        )
                    elif event.startswith("code_execution:"):
                        category = event.split(":", 1)[1]
                        _progress(
                            progress,
                            "activity",
                            message=f"Code execution — {category}",
                        )
                        if category != "success":
                            goal_result = result.get("current_goal_result", {})
                            error = goal_result.get("error") if goal_result else None
                            if error:
                                _progress(progress, "error", error=f"Error: {error}")
                    elif event.startswith("mechanical_repair:attempt_"):
                        repair_attempt = result.get(
                            "code_repair_attempts_for_current_goal", 0
                        )
                        max_repairs = result.get("max_code_repair_attempts", 50)
                        _progress(
                            progress,
                            "activity",
                            message=(
                                f"Mechanical repair: [{repair_attempt}/{max_repairs}]"
                            ),
                        )
                    elif event == "verifier:PASS":
                        _progress(progress, "activity", message="Verifier — PASS")
                        completed = result.get("completed_goal_results", [])
                        if completed:
                            _progress(
                                progress,
                                "goal_completed",
                                goal_id=str(completed[-1]["goal_id"]),
                            )
                    elif event == "verifier:REPLAN":
                        _progress(progress, "activity", message="Verifier — REPLAN")
                        _progress(
                            progress,
                            "activity",
                            message=(
                                "Scientific replan: "
                                f"[{result.get('replan_count', 0) + 1}/"
                                f"{result.get('max_replans', 1)}]"
                            ),
                        )
                trace_size = len(trace)
        if progress:
            _progress(
                progress,
                "activity",
                message=(
                    "Agent workflow — completed in "
                    f"{time.perf_counter() - workflow_started:.1f}s"
                ),
            )
        agent_directory.mkdir(parents=True, exist_ok=True)
        write_workflow_log(
            log_path=agent_directory / "workflow.log",
            result=result,
            exchanges=recorder.exchanges,
        )
        candidate = result.get("validated_final_answer")
        if candidate is not None:
            _save_json(run_directory / "candidate.json", candidate)
        workflow_status = str(result.get("status"))
        status = (
            "completed"
            if workflow_status == "completed"
            else "python_policy_failure"
            if workflow_status == "python_policy_failure"
            else "error"
        )
        error = (
            None
            if status == "completed"
            else str(result.get("policy_failure_reason") or workflow_status)
        )
        execution = json.loads(result.get("execution_result", "{}"))
        exit_code = execution.get("exit_code") if isinstance(execution, dict) else None
        timed_out = (
            bool(execution.get("timed_out", False))
            if isinstance(execution, dict)
            else False
        )
        decisions = [
            str(record["verification_decision"])
            for record in result.get("iteration_history", [])
        ]
        scripts = result.get("generated_script_count", 0)
        repairs = result.get("code_repair_count", 0)
        replans = result.get("replan_count", 0)
    except Exception as exception:
        if progress:
            _progress(
                progress,
                "workflow_failed",
                error=(
                    "Agent workflow — failed in "
                    f"{time.perf_counter() - workflow_started:.1f}s"
                ),
            )
        candidate = None
        exception_class = type(exception).__name__
        status = "infrastructure_error" if _infrastructure_error(exception) else "error"
        error = f"{type(exception).__name__}: {exception}"
        exit_code = None
        timed_out = False
        decisions = []
        scripts = repairs = replans = 0
    prompt_tokens, completion_tokens, total_tokens = _usage(recorder)
    api_call_count, retry_count = _call_counts(recorder)
    return ApproachOutcome(
        status=status,
        candidate=candidate,
        api_call_count=api_call_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        transport_retry_count=retry_count,
        execution_exit_code=exit_code,
        timed_out=timed_out,
        generated_script_count=scripts,
        local_repair_count=repairs,
        global_replan_count=replans,
        run_error=error,
        error_category=(
            "transport_api"
            if status == "infrastructure_error"
            else "python_policy"
            if status == "python_policy_failure"
            else None
        ),
        exception_class=(exception_class if status == "infrastructure_error" else None),
        verifier_decisions=decisions,
    )


def run_approach(
    *,
    approach: Approach,
    public: PublicTaskView,
    model_factory: ModelFactory,
    run_directory: Path,
    config: BenchmarkConfig,
    progress: ProgressCallback | None = None,
) -> ApproachOutcome:
    """Dispatch one clean attempt; private grading state is intentionally absent."""
    model = model_factory(approach, public)
    started = time.perf_counter()
    if approach == "direct_answer":
        outcome = run_direct_answer(
            public=public,
            model=model,
            run_directory=run_directory,
            config=config,
            progress=progress,
        )
    elif approach == "one_shot_code":
        outcome = run_one_shot_code(
            public=public,
            model=model,
            run_directory=run_directory,
            config=config,
            progress=progress,
        )
    else:
        outcome = run_agent(
            public=public,
            model=model,
            run_directory=run_directory,
            config=config,
            progress=progress,
        )
    _save_json(run_directory / "outcome.json", outcome.model_dump(mode="json"))
    (run_directory / "latency.txt").write_text(
        f"{time.perf_counter() - started:.9f}\n", encoding="utf-8"
    )
    return outcome
