"""CLI and orchestrator for the isolated three-way benchmark harness."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from data_analysis_agent.benchmark_approaches import ModelFactory, run_approach
from data_analysis_agent.benchmark_grading import grade_candidate
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
ALL_APPROACHES: list[Approach] = ["direct_answer", "one_shot_code", "agent"]


class BenchmarkError(ValueError):
    """Raised for invalid orchestration settings or benchmark output failures."""


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
    code_result_only = f"""import csv
import json

with open({data_path!r}, encoding="utf-8") as handle:
    values = [float(row["value"]) for row in csv.DictReader(handle) if row["value"]]
differences = [abs(right - left) for left, right in zip(values, values[1:])]
value = sum(differences) / len(differences)
print(json.dumps({{"mean_absolute_successive_difference": value}}))
"""
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
                    "required_outputs": ["mean_absolute_successive_difference"],
                    "constraints": [
                        "Preserve input order.",
                        "Do not impute missing values.",
                    ],
                    "success_criteria": ["Report the requested finite statistic."],
                    "depends_on": [],
                }
            ],
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
        "executor": [strategy, code_result_only] if approach == "agent" else [],
        "verifier": [verifier] if approach == "agent" else [],
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
        )
    return metrics


def run_benchmark(
    *,
    config: BenchmarkConfig,
    tasks_root: Path = DEFAULT_TASKS_ROOT,
    output_root: Path = DEFAULT_RUNS_ROOT,
    model_factory: ModelFactory | None = None,
    project_root: Path = PROJECT_ROOT,
) -> tuple[BenchmarkSummary, list[BenchmarkResult]]:
    """Run all configured approaches independently and grade externally."""
    run_id = "benchmark_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
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
                public = stage_public_task(task.public, attempt_directory)
                started = time.perf_counter()
                outcome = run_approach(
                    approach=approach,
                    public=public,
                    model_factory=factory,
                    run_directory=attempt_directory,
                    config=config,
                )
                latency = time.perf_counter() - started
                grade = grade_candidate(
                    outcome.candidate,
                    task.private,
                    candidate_error=outcome.run_error,
                )
                status = outcome.status
                if status == "completed" and not grade.passed:
                    status = "wrong_answer"
                (attempt_directory / "grade.json").write_text(
                    grade.model_dump_json(indent=2), encoding="utf-8"
                )
                result = BenchmarkResult(
                    task_id=task_id,
                    approach=approach,
                    repeat_index=repeat_index,
                    model=config.model,
                    status=status,
                    graded_success=grade.passed,
                    grader_score=grade.score,
                    grader_errors=grade.errors,
                    grader_details=grade.details,
                    api_call_count=outcome.api_call_count,
                    prompt_tokens=outcome.prompt_tokens,
                    completion_tokens=outcome.completion_tokens,
                    total_tokens=outcome.total_tokens,
                    wall_clock_latency=latency,
                    execution_exit_code=outcome.execution_exit_code,
                    timed_out=outcome.timed_out,
                    generated_script_count=outcome.generated_script_count,
                    local_repair_count=outcome.local_repair_count,
                    global_replan_count=outcome.global_replan_count,
                    final_candidate_json=outcome.candidate,
                    artifact_directory=relative_to(attempt_directory, project_root),
                    run_error=outcome.run_error,
                    not_applicable_reason=outcome.not_applicable_reason,
                    verifier_decisions=outcome.verifier_decisions,
                )
                results.append(result)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(result.model_dump_json() + "\n")

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
    lines = [
        "=" * 60,
        f"BENCHMARK — {task_label}",
        "=" * 60,
        "",
        f"Model   : {summary.config.model}",
        f"Repeats : {summary.config.repeats}",
        "",
        "Approach       Passed   API calls   Tokens   Latency   Status",
        "-" * 64,
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
            f"{metric.average_api_calls:>7.1f}   {tokens:>6}   "
            f"{metric.average_latency:>6.2f}s   {status.replace('_', ' ')}"
        )
    lines.extend(["", "-" * 60, "ERROR SUMMARY", "-" * 60])
    for approach in summary.config.approaches:
        errors = list(
            dict.fromkeys(
                error
                for result in results
                if result.approach == approach
                for error in result.grader_errors
            )
        )
        lines.append(f"{approach}:")
        lines.extend([f"- {error}" for error in errors] or ["- none"])
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", required=True, dest="task_ids")
    parser.add_argument("--approaches", type=_parse_approaches, default=ALL_APPROACHES)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=30.0)
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
            timeout_seconds=args.timeout,
            repeats=args.repeats,
            task_ids=args.task_ids,
            approaches=args.approaches,
            live=args.live,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
