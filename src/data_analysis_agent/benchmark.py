"""CLI and orchestrator for the isolated three-way benchmark harness."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import TextIO

from data_analysis_agent.benchmark_approaches import ModelFactory, run_approach
from data_analysis_agent.benchmark_grading import grade_candidate
from data_analysis_agent.benchmark_progress import BenchmarkProgressRenderer
from data_analysis_agent.benchmark_tasks import (
    BenchmarkTaskError,
    load_benchmark_task,
    reset_attempt_directory,
    stage_public_task,
)
from data_analysis_agent.benchmark_types import (
    Approach,
    ApproachMetrics,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSummary,
    PublicTaskView,
    relative_to,
)
from data_analysis_agent.config import ConfigurationError, load_settings
from data_analysis_agent.models import NebiusRoleModel, RoleModel, ScriptedRoleModel
from data_analysis_agent.nebius_client import create_nebius_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_ROOT = PROJECT_ROOT / "benchmark_tasks"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "benchmark_runs"
ALL_APPROACHES: list[Approach] = [
    "direct_answer",
    "one_shot_code",
    "agent",
    "single_agent_checker",
]


class BenchmarkError(ValueError):
    """Raised for invalid orchestration settings or benchmark output failures."""


def _slug(value: str, *, limit: int = 48) -> str:
    """Create a readable, filesystem-safe label without opaque identifiers."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (slug or "unnamed")[:limit].rstrip("-._")


def benchmark_scope_label(config: BenchmarkConfig) -> str:
    """Describe whether this is a single, three-way, or partial comparison."""
    established_three_way = {"direct_answer", "one_shot_code", "agent"}
    if set(config.approaches) == established_three_way:
        return "three_way"
    if set(config.approaches) == set(ALL_APPROACHES):
        return "four_way"
    if len(config.approaches) == 1:
        return f"{config.approaches[0]}_only"
    return "-vs-".join(config.approaches)


def build_benchmark_run_id(
    config: BenchmarkConfig, *, timestamp: str | None = None
) -> str:
    """Build a directory name that identifies tasks and comparison scope."""
    if len(config.task_ids) == 1:
        task_label = _slug(config.task_ids[0])
    else:
        task_names = "-and-".join(
            _slug(task_id, limit=32) for task_id in config.task_ids
        )
        task_label = f"{len(config.task_ids)}_tasks--{task_names}"
        if len(task_label) > 120:
            task_label = f"{len(config.task_ids)}_tasks"
    timestamp = timestamp or datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    return f"benchmark__{task_label}__{benchmark_scope_label(config)}__{timestamp}"


def _create_benchmark_run_directory(
    output_root: Path, config: BenchmarkConfig
) -> tuple[str, Path]:
    """Reserve a readable run directory, adding an ordinal only on collision."""
    base_id = build_benchmark_run_id(config)
    ordinal = 1
    while True:
        run_id = base_id if ordinal == 1 else f"{base_id}__run_{ordinal:02d}"
        run_root = output_root / run_id
        try:
            run_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            ordinal += 1
            continue
        return run_id, run_root


def _offline_model_factory(approach: Approach, public: PublicTaskView) -> RoleModel:
    """Return deterministic smoke responses without credentials or network access."""
    data_path = public.data_files[0]
    final = json.dumps(
        {
            "status": "completed",
            "answer": "The requested statistic was computed from non-missing values.",
            "key_results": {"mean_absolute_successive_difference": 2.0},
            "limitations": [],
        }
    )
    data_content = public.data_contents[data_path]
    code_result_only = f"""import csv
import io

with io.StringIO({data_content!r}) as handle:
    values = [float(row["value"]) for row in csv.DictReader(handle) if row["value"]]
differences = [abs(right - left) for left, right in zip(values, values[1:])]
value = sum(differences) / len(differences)
__agent_result__ = {{
    "status": "completed",
    "answer": "The requested statistic was computed from non-missing values.",
    "key_results": {{"mean_absolute_successive_difference": value}},
    "limitations": [],
}}
"""
    structured_code = json.dumps(
        {
            "kind": "python",
            "code": code_result_only,
            "summary": "Compute the successive-difference statistic.",
        }
    )
    single_agent_code = f"""import csv
import io

with io.StringIO({data_content!r}) as handle:
    values = [float(row["value"]) for row in csv.DictReader(handle) if row["value"]]
differences = [abs(right - left) for left, right in zip(values, values[1:])]
value = sum(differences) / len(differences)
__agent_result__ = {{
    "status": "completed",
    "answer": "The requested statistic was computed from non-missing values.",
    "key_results": {{"mean_absolute_successive_difference": value}},
    "limitations": [],
}}
"""
    single_agent_generation = json.dumps(
        {
            "kind": "python",
            "code": single_agent_code,
            "summary": "Compute the complete smoke-task answer.",
        }
    )
    final_checker = json.dumps(
        {
            "decision": "PASS",
            "repair_scope": "none",
            "feedback": "The candidate is complete.",
        }
    )
    one_shot_code = f"""import csv
import json

with open({data_path!r}, encoding="utf-8") as handle:
    values = [float(row["value"]) for row in csv.DictReader(handle) if row["value"]]
differences = [abs(right - left) for left, right in zip(values, values[1:])]
value = sum(differences) / len(differences)
print(json.dumps({{
    "status": "completed",
    "answer": "The requested statistic was computed from non-missing values.",
    "key_results": {{"mean_absolute_successive_difference": value}},
    "limitations": []
}}))
"""
    plan = json.dumps(
        {
            "scientific_objective": "Compute the requested sequence statistic.",
            "goals": [
                {
                    "goal_id": "compute_successive_difference",
                    "objective": (
                        "Compute the mean absolute successive difference of the "
                        "non-missing value sequence."
                    ),
                    "required_outputs": [
                        "status",
                        "answer",
                        "key_results.mean_absolute_successive_difference",
                        "limitations",
                    ],
                    "constraints": [
                        "Preserve input order.",
                        "Do not impute missing values.",
                    ],
                    "success_criteria": ["Report the requested finite statistic."],
                    "depends_on": [],
                }
            ],
            "final_output_goal_id": "compute_successive_difference",
        }
    )
    strategy = json.dumps(
        {
            "strategy": "generated_python",
            "capability_name": None,
            "arguments": {},
            "concise_reason": "No trusted tool computes successive differences.",
        }
    )
    verifier = json.dumps(
        {
            "decision": "PASS",
            "feedback": "The execution produced the requested statistic.",
        }
    )
    responses = {
        "direct_answer": [final] if approach == "direct_answer" else [],
        "one_shot_code": [one_shot_code] if approach == "one_shot_code" else [],
        "planner": [plan] if approach == "agent" else [],
        "executor": [strategy, structured_code] if approach == "agent" else [],
        "verifier": [verifier] if approach == "agent" else [],
        "single_agent": [single_agent_generation]
        if approach == "single_agent_checker"
        else [],
        "final_checker": [final_checker] if approach == "single_agent_checker" else [],
    }
    return ScriptedRoleModel(responses)  # type: ignore[arg-type]


def _live_model_factory(config: BenchmarkConfig) -> ModelFactory:
    settings = load_settings()
    client = create_nebius_client(settings)

    def factory(approach: Approach, public: PublicTaskView) -> RoleModel:
        del approach, public
        return NebiusRoleModel(
            client=client,
            model=config.model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_output_tokens=config.max_output_tokens,
            planner_max_output_tokens=config.planner_max_output_tokens,
            executor_max_output_tokens=config.executor_max_output_tokens,
            verifier_max_output_tokens=config.verifier_max_output_tokens,
            python_max_output_tokens=config.python_max_output_tokens,
        )

    return factory


def _metrics(results: list[BenchmarkResult]) -> dict[str, ApproachMetrics]:
    metrics: dict[str, ApproachMetrics] = {}
    approaches = list(dict.fromkeys(result.approach for result in results))
    for approach in approaches:
        selected = [result for result in results if result.approach == approach]
        attempted = [result for result in selected if result.status != "not_applicable"]
        passed = sum(result.graded_success for result in attempted)
        token_values = [
            result.total_tokens
            for result in attempted
            if result.total_tokens is not None
        ]
        denominator = len(attempted)
        metrics[approach] = ApproachMetrics(
            attempted_runs=denominator,
            passed_runs=passed,
            pass_rate=passed / denominator if denominator else 0.0,
            invalid_json_count=sum(
                result.status == "invalid_json" for result in attempted
            ),
            code_execution_failure_count=sum(
                result.status == "execution_failed" for result in attempted
            ),
            timeout_count=sum(result.timed_out for result in attempted),
            average_api_calls=(
                fmean(result.api_call_count for result in attempted)
                if attempted
                else 0.0
            ),
            average_transport_retry_count=(
                fmean(result.transport_retry_count for result in attempted)
                if attempted
                else 0.0
            ),
            average_response_retry_count=(
                fmean(result.response_retry_count for result in attempted)
                if attempted
                else 0.0
            ),
            average_total_tokens=fmean(token_values) if token_values else None,
            average_latency=(
                fmean(result.wall_clock_latency for result in attempted)
                if attempted
                else 0.0
            ),
            average_generated_script_versions=(
                fmean(result.generated_script_count for result in attempted)
                if attempted
                else 0.0
            ),
            average_local_repair_count=(
                fmean(result.local_repair_count for result in attempted)
                if attempted
                else 0.0
            ),
            average_global_replan_count=(
                fmean(result.global_replan_count for result in attempted)
                if attempted
                else 0.0
            ),
            average_global_checker_repair_count=(
                fmean(result.global_checker_repair_count for result in attempted)
                if attempted
                else 0.0
            ),
        )
    return metrics


def run_benchmark(
    *,
    config: BenchmarkConfig,
    tasks_root: Path = DEFAULT_TASKS_ROOT,
    output_root: Path = DEFAULT_RUNS_ROOT,
    model_factory: ModelFactory | None = None,
    project_root: Path = PROJECT_ROOT,
    progress_stream: TextIO | None = None,
    progress_interactive: bool | None = None,
) -> tuple[BenchmarkSummary, list[BenchmarkResult]]:
    """Run all configured approaches independently and grade externally."""
    run_id, run_root = _create_benchmark_run_directory(output_root, config)
    (run_root / "config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )
    factory = model_factory or (
        _live_model_factory(config) if config.live else _offline_model_factory
    )
    results: list[BenchmarkResult] = []
    results_path = run_root / "results.jsonl"

    for task_id in config.task_ids:
        task = load_benchmark_task(tasks_root, task_id)
        for repeat_index in range(1, config.repeats + 1):
            for approach in config.approaches:
                attempt_directory = (
                    run_root / approach / task_id / f"repeat_{repeat_index:03d}"
                )
                reset_attempt_directory(attempt_directory)
                renderer = (
                    BenchmarkProgressRenderer(
                        stream=progress_stream,
                        interactive=progress_interactive,
                        artifact_path=attempt_directory / "progress_events.jsonl",
                    )
                    if config.live and config.live_progress
                    else None
                )
                if renderer:
                    renderer.emit(
                        {
                            "type": "benchmark_started",
                            "task_id": task_id,
                            "approach": approach,
                            "model": config.model,
                            "repeat_index": repeat_index,
                            "repeats": config.repeats,
                        }
                    )
                public = stage_public_task(task.public, attempt_directory)
                started = time.perf_counter()
                outcome = run_approach(
                    approach=approach,
                    public=public,
                    model_factory=factory,
                    run_directory=attempt_directory,
                    config=config,
                    progress=renderer.emit if renderer else None,
                )
                latency = time.perf_counter() - started
                if renderer:
                    renderer.emit(
                        {
                            "type": "activity",
                            "message": f"Approach — completed in {latency:.1f}s",
                        }
                    )
                if outcome.partial_run:
                    grade = None
                    grading_skip_reason = "Intentional partial-goal smoke run."
                    if renderer:
                        renderer.emit(
                            {
                                "type": "activity",
                                "message": (
                                    "Full-task grading skipped — partial smoke run"
                                ),
                            }
                        )
                elif outcome.status == "infrastructure_error":
                    grade = None
                    grading_skip_reason = "No candidate due to infrastructure error."
                    if renderer:
                        renderer.emit(
                            {
                                "type": "activity",
                                "message": "Grading skipped — infrastructure error",
                            }
                        )
                elif outcome.status != "completed" or outcome.candidate is None:
                    grade = None
                    grading_skip_reason = (
                        "No valid task-schema candidate due to internal workflow "
                        "failure."
                    )
                    if renderer:
                        renderer.emit(
                            {
                                "type": "activity",
                                "message": (
                                    "Grading skipped — internal workflow failure"
                                ),
                            }
                        )
                else:
                    grading_skip_reason = None
                    if renderer:
                        renderer.emit(
                            {"type": "activity", "message": "Grading — starting..."}
                        )
                    grade_started = time.perf_counter()
                    grade = grade_candidate(
                        outcome.candidate,
                        task.private,
                        candidate_error=outcome.run_error,
                    )
                    if renderer:
                        renderer.emit(
                            {
                                "type": "activity",
                                "message": (
                                    "Grading — completed in "
                                    f"{time.perf_counter() - grade_started:.1f}s"
                                ),
                            }
                        )
                status = outcome.status
                if status == "completed" and grade is not None and not grade.passed:
                    status = "wrong_answer"
                grade_payload = (
                    grade.model_dump(mode="json")
                    if grade is not None
                    else (
                        {
                            "graded": False,
                            "reason": grading_skip_reason,
                            "target_reached": outcome.partial_run_reached,
                        }
                        if outcome.partial_run
                        else {
                            "graded": False,
                            "reason": grading_skip_reason,
                        }
                    )
                )
                (attempt_directory / "grade.json").write_text(
                    json.dumps(grade_payload, indent=2), encoding="utf-8"
                )
                result = BenchmarkResult(
                    task_id=task_id,
                    approach=approach,
                    repeat_index=repeat_index,
                    model=config.model,
                    status=status,
                    graded=grade is not None,
                    graded_success=grade.passed if grade is not None else False,
                    grader_score=grade.score if grade is not None else None,
                    grader_errors=grade.errors if grade is not None else [],
                    grader_details=(
                        grade.details
                        if grade is not None
                        else (
                            {
                                "error_category": "partial_smoke",
                                "target_reached": outcome.partial_run_reached,
                            }
                            if outcome.partial_run
                            else {
                                "error_category": (
                                    outcome.error_category or outcome.status
                                )
                            }
                        )
                    ),
                    api_call_count=outcome.api_call_count,
                    prompt_tokens=outcome.prompt_tokens,
                    completion_tokens=outcome.completion_tokens,
                    total_tokens=outcome.total_tokens,
                    transport_retry_count=outcome.transport_retry_count,
                    response_retry_count=outcome.response_retry_count,
                    wall_clock_latency=latency,
                    execution_exit_code=outcome.execution_exit_code,
                    timed_out=outcome.timed_out,
                    generated_script_count=outcome.generated_script_count,
                    local_repair_count=outcome.local_repair_count,
                    global_replan_count=outcome.global_replan_count,
                    global_checker_repair_count=outcome.global_checker_repair_count,
                    final_candidate_json=outcome.candidate,
                    artifact_directory=relative_to(attempt_directory, project_root),
                    run_error=outcome.run_error,
                    error_category=outcome.error_category,
                    exception_class=outcome.exception_class,
                    not_applicable_reason=outcome.not_applicable_reason,
                    verifier_decisions=outcome.verifier_decisions,
                    partial_run=outcome.partial_run,
                    partial_run_reached=outcome.partial_run_reached,
                    partial_goal_id=outcome.partial_goal_id,
                )
                results.append(result)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(result.model_dump_json() + "\n")
                if renderer:
                    renderer.emit({"type": "benchmark_finished"})
                    renderer.close()

    summary = BenchmarkSummary(
        benchmark_run_id=run_id,
        config=config,
        metrics=_metrics(results),
        results_path=relative_to(results_path, project_root),
    )
    (run_root / "summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    return summary, results


def format_benchmark_summary(
    summary: BenchmarkSummary, results: list[BenchmarkResult]
) -> str:
    """Print aggregate comparison without prompts, data, code, or private values."""
    task_label = ", ".join(summary.config.task_ids)
    if set(summary.config.approaches) == {
        "direct_answer",
        "one_shot_code",
        "agent",
    }:
        mode_label = "three-way comparison"
    elif set(summary.config.approaches) == set(ALL_APPROACHES):
        mode_label = "four-way comparison"
    elif len(summary.config.approaches) == 1:
        mode_label = "single approach"
    else:
        mode_label = "selected approach comparison"
    lines = [
        "=" * 60,
        f"BENCHMARK — {task_label}",
        "=" * 60,
        "",
        f"Run ID     : {summary.benchmark_run_id}",
        f"Mode       : {mode_label}",
        f"Approaches : {', '.join(summary.config.approaches)}",
        f"Model      : {summary.config.model}",
        f"Repeats    : {summary.config.repeats}",
        *(
            [
                "Smoke stop: after "
                f"{summary.config.stop_after_goals} verifier-PASS goal(s)"
            ]
            if summary.config.stop_after_goals is not None
            else []
        ),
        "",
        (
            "Approach       Passed   API calls   Transport retries   "
            "Response retries   Tokens   Latency   Status"
        ),
        "-" * 98,
    ]
    for approach in summary.config.approaches:
        metric = summary.metrics[approach]
        selected = [result for result in results if result.approach == approach]
        statuses = {result.status for result in selected}
        status = next(iter(statuses)) if len(statuses) == 1 else "mixed"
        tokens = (
            "n/a"
            if metric.average_total_tokens is None
            else f"{metric.average_total_tokens:.0f}"
        )
        lines.append(
            f"{approach:<15} {metric.passed_runs}/{metric.attempted_runs:<5} "
            f"{metric.average_api_calls:>7.1f}   "
            f"{metric.average_transport_retry_count:>17.1f}   "
            f"{metric.average_response_retry_count:>12.1f}   {tokens:>6}   "
            f"{metric.average_latency:>6.2f}s   {status.replace('_', ' ')}"
        )
    lines.extend(["", "-" * 60, "ERROR SUMMARY", "-" * 60])
    for approach in summary.config.approaches:
        errors = list(
            dict.fromkeys(
                error
                for result in results
                if result.approach == approach
                for error in [
                    *result.grader_errors,
                    *([result.run_error] if result.run_error else []),
                ]
            )
        )
        lines.append(f"{approach}:")
        lines.extend([f"- {error}" for error in errors] or ["- none"])
    if summary.config.stop_after_goals is not None:
        lines.extend(["", "PARTIAL SMOKE RESULTS"])
        for result in results:
            lines.append(
                f"{result.approach}/{result.task_id}: "
                f"goal={result.partial_goal_id or 'none'} "
                f"target_reached={result.partial_run_reached}"
            )
    lines.extend(
        [
            "",
            "Detailed results: "
            + summary.results_path.replace("results.jsonl", "summary.json"),
        ]
    )
    return "\n".join(lines)


def _parse_approaches(value: str) -> list[Approach]:
    if value == "all":
        return list(ALL_APPROACHES)
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in ALL_APPROACHES]
    if invalid or not values:
        raise argparse.ArgumentTypeError(
            "approaches must be all or a comma-separated subset of "
            + ",".join(ALL_APPROACHES)
        )
    return values  # type: ignore[return-value]


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", required=True, dest="task_ids")
    parser.add_argument("--approaches", type=_parse_approaches, default=ALL_APPROACHES)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--no-live-progress", action="store_false", dest="live_progress"
    )
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help=(
            "Legacy general output limit for ordinary calls; does not lower "
            "structured Python generation or repair."
        ),
    )
    parser.add_argument(
        "--planner-max-output-tokens",
        type=int,
        help="Planner output limit (default: 8192).",
    )
    parser.add_argument(
        "--executor-max-output-tokens",
        type=int,
        help="Executor strategy-selection output limit (default: 8192).",
    )
    parser.add_argument(
        "--verifier-max-output-tokens",
        type=int,
        help="Verifier output limit (default: 8192).",
    )
    parser.add_argument(
        "--python-max-output-tokens",
        type=int,
        help="Structured Python generation and repair output limit (default: 32768).",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--max-replans",
        type=_non_negative_int,
        default=1,
        help="Maximum scientific replans allowed for each agent attempt.",
    )
    parser.add_argument(
        "--stop-after-goals",
        type=int,
        help=(
            "Intentionally stop after this many verifier-PASS goals; "
            "skips full grading."
        ),
    )
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUNS_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model = args.model
        if args.live:
            settings = load_settings()
            model = model or settings.nebius_model
        model = model or "offline-scripted-smoke"
        config = BenchmarkConfig(
            model=model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
            planner_max_output_tokens=args.planner_max_output_tokens,
            executor_max_output_tokens=args.executor_max_output_tokens,
            verifier_max_output_tokens=args.verifier_max_output_tokens,
            python_max_output_tokens=args.python_max_output_tokens,
            timeout_seconds=args.timeout,
            max_replans=args.max_replans,
            repeats=args.repeats,
            task_ids=args.task_ids,
            approaches=args.approaches,
            live=args.live,
            live_progress=args.live_progress,
            stop_after_goals=args.stop_after_goals,
        )
        summary, results = run_benchmark(
            config=config,
            tasks_root=args.tasks_root,
            output_root=args.output_root,
        )
    except (BenchmarkError, BenchmarkTaskError, ConfigurationError, OSError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    print(format_benchmark_summary(summary, results))
    if args.stop_after_goals is not None:
        return int(
            not all(
                result.partial_run and result.partial_run_reached for result in results
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
