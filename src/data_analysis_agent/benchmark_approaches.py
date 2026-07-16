"""Independent direct, one-shot-code, and full-agent benchmark approaches."""

from __future__ import annotations

import csv
import io
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import JsonValue, ValidationError

from data_analysis_agent.benchmark_progress import ProgressCallback, ProgressEvent
from data_analysis_agent.benchmark_types import (
    Approach,
    ApproachOutcome,
    BenchmarkConfig,
    PublicTaskView,
)
from data_analysis_agent.config import (
    code_repair_settings,
    full_agent_reliability_settings,
)
from data_analysis_agent.demo import write_workflow_log
from data_analysis_agent.final_output import DeterministicFinalOutputProvider
from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import (
    ProviderResponseError,
    RecordingRoleModel,
    RoleModel,
)
from data_analysis_agent.prompts import build_python_repair_messages
from data_analysis_agent.public_schema import (
    validate_against_public_schema as _validate_against_public_schema,
)
from data_analysis_agent.python_runner import LocalPythonRunner, PythonExecutionResult
from data_analysis_agent.schemas import (
    FinalCheckerOutput,
    IntermediateGoal,
    PythonGeneration,
    PythonRepair,
)
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

SINGLE_AGENT_SYSTEM_PROMPT = """You are a single iterative data-analysis agent.
Solve the complete public task by returning one PythonGeneration JSON object. Its
code_lines must be one physical, comment-free Python source line each. Read only
the stated staged paths, write only inside the assigned execution directory, and
assign the complete public answer-schema object to module-level __agent_result__.
Every staged path must be quoted inside a valid Python expression; never emit a
bare path as a code line. The result must contain only JSON-compatible values.
Use concise code; a trusted runner owns result serialization. Mechanical failures
may receive a bounded repair,
but there is no Planner or per-goal Verifier in this approach. Use the semantic
column names specified by the task; never substitute a positional first-column
selection unless the task explicitly requests it. When reporting rows removed by
filtering or deduplication, record the count before applying that transformation.
For exact duplicate removal, use len(original_rows) - len(deduplicated_rows), then
continue analysis with the deduplicated rows."""

FINAL_CHECKER_SYSTEM_PROMPT = """You are an independent final completeness
checker. Assess the public task, public answer schema, candidate answer, factual
execution result, artifact summary, and limitations. Return exactly one
FinalCheckerOutput JSON object. Return PASS only if the candidate is complete and
supported by an independent cross-check of the supplied public data; matching
execution output alone is not evidence of correctness. Verify explicit sequential
rules and removal counts from the raw input before passing. For exact duplicate
removal, independently check raw_row_count - deduplicated_row_count. Return REPAIR with
concise actionable feedback and repair_scope="format_only" for representation-only
issues or
"rerun_analysis" for any scientific/data-analysis correction. You do not write
code and are called exactly once."""


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


def build_single_agent_messages(
    public: PublicTaskView, run_directory: Path, staged_files: list[Path]
) -> list[dict[str, str]]:
    """Give the ablation's sole analysis agent the same public task boundary."""
    return [
        {"role": "system", "content": SINGLE_AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task ID: {public.task_id}\n\nTask prompt:\n{public.prompt}\n\n"
                f"Public answer schema:\n{json.dumps(public.answer_schema)}\n\n"
                "Process current working directory (cwd):\n"
                f"{run_directory.resolve()}\n\n"
                "Allowed staged input paths (absolute paths; use these exact paths "
                "directly as Python string literals):\n"
                f"{json.dumps([str(path) for path in staged_files])}\n\n"
                "Relative input paths are invalid. Relative output paths resolve "
                "inside the assigned execution directory.\n\n"
                "Return the complete answer-schema object in __agent_result__."
            ),
        },
    ]


def build_final_checker_messages(
    *,
    public: PublicTaskView,
    candidate: dict[str, JsonValue],
    execution: dict[str, JsonValue],
) -> list[dict[str, str]]:
    """Keep final checking independent from the single-agent role history."""
    input_sections = []
    for name in public.data_files:
        key = _content_key(public, name)
        input_sections.append(
            f"----- BEGIN FILE: {Path(name).name} -----\n"
            f"{public.data_contents[key].rstrip()}\n"
            f"----- END FILE: {Path(name).name} -----"
        )
    return [
        {"role": "system", "content": FINAL_CHECKER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task prompt:\n{public.prompt}\n\n"
                f"Public answer schema:\n{json.dumps(public.answer_schema)}\n\n"
                "Public input data (cross-check numerical claims against these "
                "files):\n" + "\n\n".join(input_sections) + "\n\n"
                f"Final candidate answer:\n{json.dumps(candidate)}\n\n"
                f"Factual execution result:\n{json.dumps(execution)}\n\n"
                "Artifacts and limitations: no unregistered artifacts; the candidate's "
                "limitations field is authoritative."
            ),
        },
    ]


def build_final_answer_repair_messages(
    *, public: PublicTaskView, candidate: dict[str, JsonValue], feedback: str
) -> list[dict[str, str]]:
    """Request the one allowed global answer-only repair from the analysis model."""
    return [
        {
            "role": "system",
            "content": (
                "Repair one final JSON answer. Return exactly one JSON object "
                "matching the public answer schema, without Markdown or explanation. "
                "This is the "
                "only global answer repair; do not return Python."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task prompt:\n{public.prompt}\n\n"
                f"Public answer schema:\n{json.dumps(public.answer_schema)}\n\n"
                f"Candidate to repair:\n{json.dumps(candidate)}\n\n"
                f"Independent checker feedback:\n{feedback}"
            ),
        },
    ]


def build_single_agent_checker_repair_messages(
    *,
    public: PublicTaskView,
    source: str,
    execution: PythonExecutionResult,
    feedback: str,
    repair_scope: str,
    staged_files: list[Path],
    execution_directory: Path,
) -> list[dict[str, str]]:
    """Request a checker-directed repair that is grounded in a second execution."""
    return [
        {
            "role": "system",
            "content": (
                "Regenerate executable analysis Python, not an answer-only JSON "
                "rewrite. The code_lines replace the entire previous program: "
                "return a complete standalone program, never a patch, diff, or "
                "changed-line fragment. "
                "Return exactly one PythonRepair JSON object. The repaired program is "
                "executed and its __agent_result__ is the only candidate that can "
                "be used."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original task:\n{public.prompt}\n\n"
                f"Exact public answer schema:\n{json.dumps(public.answer_schema)}\n\n"
                "Process cwd / assigned execution directory:\n"
                f"{execution_directory.resolve()}\n\n"
                "Allowed absolute staged input paths (use directly as string literals; "
                "relative input paths are invalid):\n"
                f"{json.dumps([str(path) for path in staged_files])}\n\n"
                f"Previous generated Python:\n{source}\n\n"
                "Previous factual execution result:\n"
                f"{json.dumps(execution.model_dump(mode='json'))}\n\n"
                f"Checker repair scope: {repair_scope}\n"
                f"Checker feedback: {feedback}\n\n"
                "Use kind='python_repair' and addressed_failure_category="
                "'result_contract_error'."
            ),
        },
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


def _call_counts(recorder: RecordingRoleModel) -> tuple[int, int, int]:
    return (
        sum(exchange.api_request_count for exchange in recorder.exchanges),
        sum(exchange.transport_retry_count for exchange in recorder.exchanges),
        sum(exchange.response_retry_count for exchange in recorder.exchanges),
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
            "single_agent": "Single agent",
            "final_checker": "Final checker",
        }
        label = labels[role]
        if phase == "start":
            message = (
                "Planner — started; generating plan steps..."
                if role == "planner"
                else f"{label} — calling model..."
            )
            _progress(progress, "activity", message=message)
        elif error:
            _progress(
                progress,
                "error",
                error=f"{label} — failed after {elapsed:.1f}s: {error}",
            )
        else:
            message = (
                f"Planner — plan generated in {elapsed:.1f}s"
                if role == "planner"
                else f"{label} — completed in {elapsed:.1f}s"
            )
            _progress(
                progress,
                "activity",
                message=message,
            )

    return observe


def _infrastructure_error(exception: Exception) -> bool:
    if isinstance(exception, ProviderResponseError):
        return True
    name = type(exception).__name__
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
    } or (name == "InternalServerError")


def _provider_response_error(exception_class: str | None) -> bool:
    return exception_class in {
        "EmptyModelResponseError",
        "MalformedModelResponseError",
        "ModelOutputLimitError",
        "ModelRefusalError",
    }


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
    api_call_count, retry_count, response_retry_count = _call_counts(recorder)
    return ApproachOutcome(
        status=status,
        candidate=candidate,
        api_call_count=api_call_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        transport_retry_count=retry_count,
        response_retry_count=response_retry_count,
        run_error=error,
        error_category=(
            "provider_response"
            if _provider_response_error(exception_class)
            else "transport_api"
            if status == "infrastructure_error"
            else None
        ),
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
            result_mode="legacy_stdout",
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
            elif (execution.error or "").startswith("ResultContractError"):
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
    api_call_count, retry_count, response_retry_count = _call_counts(recorder)
    return ApproachOutcome(
        status=status,
        candidate=candidate,
        api_call_count=api_call_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        transport_retry_count=retry_count,
        response_retry_count=response_retry_count,
        execution_exit_code=execution.exit_code if execution else None,
        timed_out=execution.timed_out if execution else False,
        generated_script_count=1 if execution else 0,
        run_error=error,
        error_category=(
            "provider_response"
            if _provider_response_error(exception_class)
            else "transport_api"
            if status == "infrastructure_error"
            else "python_policy"
            if status == "python_policy_failure"
            else None
        ),
        exception_class=(exception_class if status == "infrastructure_error" else None),
    )


def _single_agent_status(execution: PythonExecutionResult | None) -> str:
    """Map shared runner facts to the benchmark's public attempt status."""
    if execution is None:
        return "error"
    if execution.timed_out:
        return "timed_out"
    if (execution.error or "").startswith("PythonPolicyError:"):
        return "python_policy_failure"
    return "execution_failed"


@dataclass
class _SingleAgentCoreResult:
    """Facts from the shared iterative analysis phase before any final checker."""

    recorder: RecordingRoleModel
    execution_directory: Path
    staged_files: list[Path]
    runner: LocalPythonRunner
    execution: PythonExecutionResult | None
    source: str
    candidate: dict[str, JsonValue] | None
    status: str
    error: str | None
    exception_class: str | None
    local_repairs: int


def _run_iterative_single_agent_core(
    *,
    public: PublicTaskView,
    model: RoleModel,
    run_directory: Path,
    config: BenchmarkConfig,
    progress: ProgressCallback | None,
) -> _SingleAgentCoreResult:
    """Run the one analysis generator plus bounded mechanical code repair.

    This deliberately ends at deterministic public-schema validation.  Low and
    middle architectures call this exact function, making their pre-checker
    generation, paths, runner, and repair behavior identical.
    """
    recorder = RecordingRoleModel(model, _model_observer(progress))
    execution_directory = run_directory / "single_agent_run"
    staged_files = _resolved_public_files(public, run_directory)
    goal = IntermediateGoal(
        goal_id="single_analysis",
        objective="Produce the complete public answer-schema object.",
        required_outputs=["complete answer-schema object"],
        constraints=["Use only the staged public files."],
        success_criteria=["The answer is valid and JSON-compatible."],
    )
    runner = LocalPythonRunner(
        timeout_seconds=config.timeout_seconds,
        progress_callback=(
            lambda message: (
                _progress(progress, "activity", message=message) if progress else None
            )
        ),
    )
    max_repairs, no_progress_limit = code_repair_settings()
    execution: PythonExecutionResult | None = None
    source = ""
    local_repairs = 0
    same_family = 0
    previous_family: str | None = None
    error: str | None = None
    exception_class: str | None = None
    candidate: dict[str, JsonValue] | None = None
    status = "execution_failed"
    try:
        for version in range(1, max_repairs + 2):
            if version == 1:
                raw = recorder.generate_structured(
                    role="single_agent",
                    messages=build_single_agent_messages(
                        public, execution_directory, staged_files
                    ),
                    schema_name="single_agent_python_generation",
                    schema=PythonGeneration.model_json_schema(),
                )
                contract_class: type[PythonGeneration] | type[PythonRepair] = (
                    PythonGeneration
                )
                artifact_name = "python_generation"
            else:
                local_repairs += 1
                previous = execution
                raw = recorder.generate_structured(
                    role="single_agent",
                    messages=build_python_repair_messages(
                        current_goal=goal.model_dump(mode="json"),
                        code=source,
                        failure_category=(
                            previous.failure_category
                            if previous
                            else "generation_contract_error"
                        )
                        or "generation_contract_error",
                        stdout=previous.stdout if previous else "",
                        stderr=previous.stderr if previous else "",
                        error=previous.error if previous else error,
                        staged_file_paths=[str(path) for path in staged_files],
                        goal_directory=str(execution_directory),
                        input_context=_agent_input_context(public),
                        repair_history=[],
                    ),
                    schema_name="single_agent_python_repair",
                    schema=PythonRepair.model_json_schema(),
                )
                contract_class = PythonRepair
                artifact_name = "python_repair"
            raw_path = execution_directory / f"{artifact_name}_v{version}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(raw, encoding="utf-8")
            try:
                contract = contract_class.model_validate_json(raw)
            except ValidationError as validation_error:
                error = f"GenerationContractError: {validation_error}"
                execution = None
                family = "generation_contract_error"
            else:
                source = contract.source()
                execution = runner.run(
                    code=source,
                    goal_directory=execution_directory,
                    allowed_files=staged_files,
                    version=version,
                )
                if execution.success:
                    candidate = execution.result
                    _validate_against_public_schema(candidate, public.answer_schema)
                    status = "completed"
                    error = None
                    break
                error = execution.error
                family = execution.failure_category or "runtime_error"
            same_family = same_family + 1 if family == previous_family else 1
            previous_family = family
            if same_family >= no_progress_limit or version > max_repairs:
                break
        if status != "completed":
            status = _single_agent_status(execution)
            if execution is None:
                status = "execution_failed"
    except (ValueError, ValidationError) as exception:
        candidate = None
        status = "invalid_json"
        error = f"{type(exception).__name__}: {exception}"
    except Exception as exception:
        candidate = None
        exception_class = type(exception).__name__
        status = "infrastructure_error" if _infrastructure_error(exception) else "error"
        error = f"{type(exception).__name__}: {exception}"
    return _SingleAgentCoreResult(
        recorder=recorder,
        execution_directory=execution_directory,
        staged_files=staged_files,
        runner=runner,
        execution=execution,
        source=source,
        candidate=candidate,
        status=status,
        error=error,
        exception_class=exception_class,
        local_repairs=local_repairs,
    )


def _single_agent_outcome(
    core: _SingleAgentCoreResult,
    *,
    checker_decisions: list[str] | None = None,
    checker_repairs: int = 0,
) -> ApproachOutcome:
    """Convert shared-core facts to an externally reportable outcome."""
    prompt_tokens, completion_tokens, total_tokens = _usage(core.recorder)
    api_call_count, retry_count, response_retry_count = _call_counts(core.recorder)
    return ApproachOutcome(
        status=core.status,
        candidate=core.candidate,
        api_call_count=api_call_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        transport_retry_count=retry_count,
        response_retry_count=response_retry_count,
        execution_exit_code=core.execution.exit_code if core.execution else None,
        timed_out=core.execution.timed_out if core.execution else False,
        generated_script_count=len(
            list(core.execution_directory.glob("generated_code_v*.py"))
        ),
        local_repair_count=core.local_repairs,
        global_replan_count=0,
        global_checker_repair_count=checker_repairs,
        run_error=core.error,
        error_category=(
            "provider_response"
            if _provider_response_error(core.exception_class)
            else "transport_api"
            if core.status == "infrastructure_error"
            else "python_policy"
            if core.status == "python_policy_failure"
            else None
        ),
        exception_class=(
            core.exception_class if core.status == "infrastructure_error" else None
        ),
        verifier_decisions=checker_decisions or [],
    )


def run_single_agent_checker(
    *,
    public: PublicTaskView,
    model: RoleModel,
    run_directory: Path,
    config: BenchmarkConfig,
    progress: ProgressCallback | None = None,
) -> ApproachOutcome:
    """Run one iterative code agent and exactly one independent final checker."""
    core = _run_iterative_single_agent_core(
        public=public,
        model=model,
        run_directory=run_directory,
        config=config,
        progress=progress,
    )
    if core.status != "completed" or core.candidate is None or core.execution is None:
        return _single_agent_outcome(core)
    checker_decisions: list[str] = []
    checker_repairs = 0
    try:
        checker_raw = core.recorder.generate_structured(
            role="final_checker",
            messages=build_final_checker_messages(
                public=public,
                candidate=core.candidate,
                execution=core.execution.model_dump(mode="json"),
            ),
            schema_name="single_agent_final_checker",
            schema=FinalCheckerOutput.model_json_schema(),
        )
        (core.execution_directory / "final_checker_response.json").write_text(
            checker_raw, encoding="utf-8"
        )
        checker = FinalCheckerOutput.model_validate_json(checker_raw)
        checker_decisions = [checker.decision]
        if checker.decision == "REPAIR":
            checker_repairs = 1
            repaired_raw = core.recorder.generate_structured(
                role="single_agent",
                messages=build_single_agent_checker_repair_messages(
                    public=public,
                    source=core.source,
                    execution=core.execution,
                    feedback=checker.feedback,
                    repair_scope=checker.repair_scope,
                    staged_files=core.staged_files,
                    execution_directory=core.execution_directory,
                ),
                schema_name="single_agent_checker_python_repair",
                schema=PythonRepair.model_json_schema(),
            )
            (core.execution_directory / "final_checker_python_repair.json").write_text(
                repaired_raw, encoding="utf-8"
            )
            repaired = PythonRepair.model_validate_json(repaired_raw)
            core.source = repaired.source()
            core.execution = core.runner.run(
                code=core.source,
                goal_directory=core.execution_directory,
                allowed_files=core.staged_files,
                version=core.local_repairs + 2,
            )
            if not core.execution.success:
                core.candidate = None
                core.status = _single_agent_status(core.execution)
                core.error = core.execution.error
            else:
                core.candidate = core.execution.result
                _validate_against_public_schema(core.candidate, public.answer_schema)
        if core.execution.success:
            _save_json(run_directory / "candidate.json", core.candidate)
            core.status = "completed"
            core.error = None
    except (ValueError, ValidationError) as exception:
        core.candidate = None
        core.status = "invalid_json"
        core.error = f"{type(exception).__name__}: {exception}"
    except Exception as exception:
        core.candidate = None
        core.exception_class = type(exception).__name__
        core.status = (
            "infrastructure_error" if _infrastructure_error(exception) else "error"
        )
        core.error = f"{type(exception).__name__}: {exception}"
    return _single_agent_outcome(
        core, checker_decisions=checker_decisions, checker_repairs=checker_repairs
    )


def run_single_agent(
    *,
    public: PublicTaskView,
    model: RoleModel,
    run_directory: Path,
    config: BenchmarkConfig,
    progress: ProgressCallback | None = None,
) -> ApproachOutcome:
    """Run the low architecture: shared analysis core with no model checker."""
    core = _run_iterative_single_agent_core(
        public=public,
        model=model,
        run_directory=run_directory,
        config=config,
        progress=progress,
    )
    if core.status == "completed" and core.candidate is not None:
        _save_json(run_directory / "candidate.json", core.candidate)
    return _single_agent_outcome(core)


def _profile_scalar_type(values: list[str]) -> str:
    """Infer a deliberately small, model-facing scalar type."""
    present = [value for value in values if value.strip()]
    if not present:
        return "unknown"
    try:
        for value in present:
            int(value)
        return "integer"
    except ValueError:
        pass
    try:
        for value in present:
            float(value)
        return "number"
    except ValueError:
        return "string"


def _agent_input_profile(public: PublicTaskView) -> dict[str, JsonValue]:
    """Summarize public inputs without repeating complete datasets in prompts."""
    files: list[dict[str, JsonValue]] = []
    for path in public.data_files:
        key = _content_key(public, path)
        content = public.data_contents[key]
        if Path(path).suffix.lower() != ".csv":
            files.append(
                {
                    "filename": path,
                    "format": Path(path).suffix.lower().lstrip(".") or "text",
                    "character_count": len(content),
                    "preview": content[:500],
                }
            )
            continue
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        columns = list(reader.fieldnames or [])
        files.append(
            {
                "filename": path,
                "format": "csv",
                "row_count": len(rows),
                "columns": [
                    {
                        "name": column,
                        "inferred_type": _profile_scalar_type(
                            [str(row.get(column, "")) for row in rows]
                        ),
                        "missing_count": sum(
                            not str(row.get(column, "")).strip() for row in rows
                        ),
                    }
                    for column in columns
                ],
                "preview": [
                    {column: row.get(column, "") for column in columns}
                    for row in rows[:3]
                ],
            }
        )
    return {"files": files}


def _agent_input_context(public: PublicTaskView) -> str:
    return json.dumps(_agent_input_profile(public), ensure_ascii=False, indent=2)


def run_agent(
    *,
    public: PublicTaskView,
    model: RoleModel,
    run_directory: Path,
    config: BenchmarkConfig,
    progress: ProgressCallback | None = None,
) -> ApproachOutcome:
    """Invoke the existing full Planner/Executor/Verifier workflow from scratch."""
    reliability = full_agent_reliability_settings()
    recorder = RecordingRoleModel(
        model,
        _model_observer(progress),
        max_calls=reliability["max_model_calls"],
    )
    agent_directory = run_directory / "agent_run"
    exception_class = None
    workflow_started = time.perf_counter()
    result: AgentState = {}
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
            "input_context": _agent_input_context(public),
            "input_profile": _agent_input_profile(public),
            "answer_schema": public.answer_schema,
            **reliability,
            "contract_escalated_goal_ids": [],
            "rollback_count": 0,
            "run_directory": str(agent_directory),
            "replan_count": 0,
            "max_replans": config.max_replans,
            "stop_after_goals": config.stop_after_goals,
            "output_repair_count": 0,
            "max_output_repairs": 1,
            "output_validation_history": [],
            "trace": [],
            "iteration_history": [],
        }
        result = initial_state
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
                        completed_goal_ids = [
                            str(item["goal_id"])
                            for item in result.get("completed_goal_results", [])
                        ]
                        current_index = result.get("current_goal_index", 0)
                        current_goal_id = (
                            str(goals[current_index]["goal_id"])
                            if current_index < len(goals)
                            else None
                        )
                        _progress(
                            progress,
                            "plan_available",
                            goals=goals,
                            completed_goal_ids=completed_goal_ids,
                            current_goal_id=current_goal_id,
                            scientific_replan_count=result.get("replan_count", 0),
                            plan_revision=result.get("plan_revision", 0),
                            is_scientific_replan=result.get(
                                "is_scientific_replan", False
                            ),
                            total_goal_count=result.get("total_goal_count", len(goals)),
                            preserved_completed_count=result.get(
                                "preserved_completed_count", len(completed_goal_ids)
                            ),
                            remaining_goal_count=result.get(
                                "remaining_goal_count",
                                len(goals) - len(completed_goal_ids),
                            ),
                            invalidated_goal_ids=result.get("invalidated_goal_ids", []),
                            new_goal_ids=result.get("new_goal_ids", []),
                        )
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
                        replan_count = result.get("replan_count", 0)
                        max_replans = result.get("max_replans", 1)
                        if replan_count < max_replans:
                            message = (
                                f"Scientific replan: [{replan_count + 1}/{max_replans}]"
                            )
                        else:
                            message = "Scientific replan budget exhausted"
                        _progress(progress, "activity", message=message)
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
        partial_run = config.stop_after_goals is not None
        partial_reached = bool(result.get("partial_run_reached", False))
        completed_goals = list(result.get("completed_goal_results", []))
        partial_goal_id = (
            str(completed_goals[-1].get("goal_id")) if completed_goals else None
        )
        if partial_run:
            partial_summary = {
                "requested_stop_after_goals": config.stop_after_goals,
                "target_reached": partial_reached,
                "goal_id": partial_goal_id,
                "execution_status": (
                    completed_goals[-1].get("success") if completed_goals else False
                ),
                "verifier_decision": result.get("verification_decision"),
                "repair_counts": {
                    "mechanical": result.get("code_repair_count", 0),
                    "scientific_replans": result.get("replan_count", 0),
                },
                "result_artifacts": (
                    completed_goals[-1].get("artifact_paths", [])
                    if completed_goals
                    else []
                ),
                "run_id": result.get("run_id"),
            }
            _save_json(agent_directory / "partial_run_summary.json", partial_summary)
        candidate = None if partial_run else result.get("validated_final_answer")
        if candidate is not None:
            _save_json(run_directory / "candidate.json", candidate)
        workflow_status = str(result.get("status"))
        status = (
            "completed"
            if workflow_status in {"completed", "completed_with_limitations"}
            or partial_reached
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
        result = {
            **result,
            "status": status,
            "final_answer": error,
        }
        exit_code = None
        timed_out = False
        decisions = [
            str(record["verification_decision"])
            for record in result.get("iteration_history", [])
            if "verification_decision" in record
        ]
        scripts = int(result.get("generated_script_count", 0))
        repairs = int(result.get("code_repair_count", 0))
        replans = int(result.get("replan_count", 0))
        partial_run = config.stop_after_goals is not None
        partial_reached = False
        partial_goal_id = None
    finally:
        if result:
            try:
                agent_directory.mkdir(parents=True, exist_ok=True)
                write_workflow_log(
                    log_path=agent_directory / "workflow.log",
                    result=result,
                    exchanges=recorder.exchanges,
                )
            except Exception:
                pass
    prompt_tokens, completion_tokens, total_tokens = _usage(recorder)
    api_call_count, retry_count, response_retry_count = _call_counts(recorder)
    return ApproachOutcome(
        status=status,
        candidate=candidate,
        api_call_count=api_call_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        transport_retry_count=retry_count,
        response_retry_count=response_retry_count,
        execution_exit_code=exit_code,
        timed_out=timed_out,
        generated_script_count=scripts,
        local_repair_count=repairs,
        global_replan_count=replans,
        run_error=error,
        error_category=(
            "provider_response"
            if _provider_response_error(exception_class)
            else "transport_api"
            if status == "infrastructure_error"
            else "python_policy"
            if status == "python_policy_failure"
            else None
        ),
        exception_class=(exception_class if status == "infrastructure_error" else None),
        verifier_decisions=decisions,
        partial_run=partial_run,
        partial_run_reached=partial_reached,
        partial_goal_id=partial_goal_id,
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
    elif approach == "single_agent":
        outcome = run_single_agent(
            public=public,
            model=model,
            run_directory=run_directory,
            config=config,
            progress=progress,
        )
    elif approach == "single_agent_checker":
        outcome = run_single_agent_checker(
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
