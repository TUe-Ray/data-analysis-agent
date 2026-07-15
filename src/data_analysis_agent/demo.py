"""CLI demos and live Verifier evaluation for Prototype V0."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from openai import OpenAIError

from data_analysis_agent.config import ConfigurationError, load_settings
from data_analysis_agent.evaluation import (
    DIVIDER,
    SUBDIVIDER,
    EvaluationFixtureError,
    EvaluationOutputError,
    format_evaluation_summary,
    load_verifier_cases,
    run_verifier_evaluation,
)
from data_analysis_agent.final_output import (
    DeterministicFinalOutputProvider,
    build_scripted_output_provider,
)
from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import (
    ModelExchange,
    NebiusRoleModel,
    RecordingRoleModel,
    RoleModel,
    build_scripted_model,
)
from data_analysis_agent.nebius_client import create_nebius_client
from data_analysis_agent.nodes import VerifierOutputError
from data_analysis_agent.state import AgentState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
MAX_FILE_BYTES = 50 * 1024
SUPPORTED_SUFFIXES = {".csv", ".txt"}


class DemoInputError(ValueError):
    """Raised when a demo input cannot be safely staged."""


def _read_small_text_file(path: Path) -> str:
    if not path.is_file():
        raise DemoInputError(f"Input file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise DemoInputError(f"Unsupported input file type: {path.suffix or '(none)'}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DemoInputError(f"Input file exceeds the 50 KB limit: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise DemoInputError(f"Input file is not valid UTF-8 text: {path}") from error


def stage_input_files(paths: list[Path]) -> tuple[list[str], str]:
    """Read small text/CSV inputs into one explicit, named model context."""
    if not paths:
        raise DemoInputError("At least one input file is required")
    names: list[str] = []
    sections: list[str] = []
    for path in paths:
        content = _read_small_text_file(path)
        names.append(path.name)
        sections.append(f"File: {path.name}\n{content.rstrip()}")
    return names, "\n\n".join(sections)


def _offline_inputs(scenario: str) -> tuple[Path, list[Path]]:
    if scenario == "happy":
        return (
            PROJECT_ROOT / "examples/prompts/happy_path.txt",
            [PROJECT_ROOT / "examples/data/simple_measurements.csv"],
        )
    if scenario in {
        "replan",
        "max-replan",
        "valid-json",
        "output-repair",
        "malformed-json",
        "output-failure",
        "trusted-tools-success",
    }:
        return (
            PROJECT_ROOT / "examples/prompts/verifier_trap.txt",
            [PROJECT_ROOT / "examples/data/measurements_with_missing.csv"],
        )
    if scenario in {
        "generated-python-success",
        "generated-python-repair",
        "generated-python-failure",
    }:
        return (
            PROJECT_ROOT / "examples/prompts/successive_difference.txt",
            [PROJECT_ROOT / "examples/data/measurements_with_missing.csv"],
        )
    raise DemoInputError(f"Unknown offline scenario: {scenario}")


def _create_demo_log_path(output_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run_directory = output_dir / f"demo_{timestamp}"
    try:
        run_directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise DemoInputError(
            f"Could not create demo output directory {run_directory}: {error}"
        ) from error
    return run_directory / "workflow.log"


def _write_workflow_log(
    *, log_path: Path, result: AgentState, exchanges: list[ModelExchange]
) -> None:
    lines = [
        "PROTOTYPE V0 — DETAILED WORKFLOW LOG",
        f"Timestamp: {datetime.now(UTC).isoformat()}",
        f"Question:\n{result['question']}",
        f"Staged files: {', '.join(result['file_paths'])}",
        f"Staged input context:\n{result['input_context']}",
        f"Trace: {' -> '.join(result['trace'])}",
        "Structured high-level plan:",
        json.dumps(result.get("high_level_plan"), indent=2, ensure_ascii=False),
        "Capability catalog:",
        json.dumps(result.get("capability_catalog", []), indent=2),
        "Completed GoalResults:",
        json.dumps(result.get("completed_goal_results", []), indent=2),
        f"Run artifact directory: {result.get('run_directory', 'none')}",
        "",
    ]
    for index, exchange in enumerate(exchanges, start=1):
        lines.extend(
            [
                SUBDIVIDER,
                f"Exchange: {index}",
                f"Role: {exchange.role}",
                "Exact messages:",
                json.dumps(exchange.messages, indent=2, ensure_ascii=False),
                "Raw response:",
                exchange.response or "none",
                f"Latency seconds: {exchange.latency_seconds:.6f}",
                f"Error: {exchange.error or 'none'}",
                "",
            ]
        )
    lines.extend(
        [
            SUBDIVIDER,
            "Iteration history:",
            json.dumps(result.get("iteration_history", []), indent=2),
            "Raw final-answer-generator output:",
            result.get("raw_final_output", "none"),
            "Pydantic parse result:",
            json.dumps(
                result.get("validated_final_answer"), indent=2, ensure_ascii=False
            ),
            "Output validation status:",
            str(result.get("output_validation_status")),
            "Output validation errors:",
            result.get("output_validation_error", ""),
            "Raw repair output:",
            result.get("raw_repair_output", "none"),
            "Output validation history:",
            json.dumps(result.get("output_validation_history", []), indent=2),
            f"Repair count: {result.get('output_repair_count', 0)}",
            "Final validated JSON:",
            json.dumps(
                result.get("validated_final_answer"), indent=2, ensure_ascii=False
            ),
            f"Final status: {result['status']}",
            f"Final answer:\n{result['final_answer']}",
        ]
    )
    try:
        log_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as error:
        raise DemoInputError(
            f"Could not write workflow log {log_path}: {error}"
        ) from error


def run_demo(
    *,
    mode: Literal["offline", "live"],
    scenario: str = "happy",
    prompt_path: Path | None = None,
    file_paths: list[Path] | None = None,
    max_replans: int = 1,
    log_path: Path | None = None,
) -> AgentState:
    """Stage inputs and invoke the real graph in offline or live mode."""
    if max_replans < 0:
        raise DemoInputError("max_replans must be zero or greater")

    model: RoleModel
    if mode == "offline":
        default_prompt, default_files = _offline_inputs(scenario)
        prompt_path = prompt_path or default_prompt
        file_paths = file_paths or default_files
        model = build_scripted_model(scenario)
        output_provider = build_scripted_output_provider(scenario)
    elif mode == "live":
        if prompt_path is None or not file_paths:
            raise DemoInputError("Live mode requires --prompt and at least one --file")
        settings = load_settings()
        model = NebiusRoleModel(
            client=create_nebius_client(settings),
            model=settings.nebius_model,
        )
        output_provider = DeterministicFinalOutputProvider()
    else:
        raise DemoInputError(f"Unknown demo mode: {mode}")

    question = _read_small_text_file(prompt_path).strip()
    staged_names, input_context = stage_input_files(file_paths)
    staged_paths = [str(path.resolve()) for path in file_paths]
    recorder = RecordingRoleModel(model)
    result = cast(
        AgentState,
        build_graph(recorder, output_provider).invoke(
            {
                "question": question,
                "file_paths": staged_names,
                "staged_file_paths": staged_paths,
                "input_context": input_context,
                "run_directory": str(log_path.parent) if log_path else "",
                "replan_count": 0,
                "max_replans": max_replans,
                "output_repair_count": 0,
                "max_output_repairs": 1,
                "output_validation_history": [],
                "trace": [],
                "iteration_history": [],
            }
        ),
    )
    if log_path is not None:
        _write_workflow_log(
            log_path=log_path, result=result, exchanges=recorder.exchanges
        )
    return result


def format_workflow_result(
    *, mode: str, scenario: str, result: AgentState, log_path: Path
) -> str:
    """Format concise iteration-level output without complete model messages."""
    demo_name = "LIVE" if mode == "live" else scenario.upper().replace("-", " ")
    lines = [
        DIVIDER,
        f"PROTOTYPE V0 — {demo_name} DEMO",
        DIVIDER,
        "",
        "Question:",
        result["question"],
        "",
        "Staged files:",
        *[f"- {name}" for name in result["file_paths"]],
    ]
    for record in result.get("iteration_history", []):
        goal_id = record.get("goal_id")
        if goal_id:
            plan = result.get("high_level_plan", {})
            goals = plan.get("goals", []) if isinstance(plan, dict) else []
            goal = next(
                (
                    item
                    for item in goals
                    if isinstance(item, dict) and item.get("goal_id") == goal_id
                ),
                {},
            )
            goal_number = next(
                (
                    index
                    for index, item in enumerate(goals, start=1)
                    if isinstance(item, dict) and item.get("goal_id") == goal_id
                ),
                record["iteration"],
            )
            capability = record.get("capability_name")
            lines.extend(
                [
                    "",
                    SUBDIVIDER,
                    f"GOAL {goal_number} OF {len(goals)} — {goal_id}",
                    SUBDIVIDER,
                    "Objective:",
                    str(goal.get("objective", "")),
                    "",
                    "Strategy:",
                    (f"{record.get('strategy')} — {capability or ''}").rstrip(" —"),
                    "",
                    "Result summary:",
                    record["execution_result"],
                    "",
                    "Verifier:",
                    f"Decision : {record['verification_decision']}",
                    f"Feedback : {record['verification_feedback']}",
                    "",
                    "Route:",
                    record["route"],
                ]
            )
            continue
        lines.extend(
            [
                "",
                SUBDIVIDER,
                f"ITERATION {record['iteration']}",
                SUBDIVIDER,
                "Plan:",
                record["plan"],
                "",
                "Execution result:",
                record["execution_result"],
                "",
                "Verifier:",
                f"Decision : {record['verification_decision']}",
                f"Feedback : {record['verification_feedback']}",
                "",
                "Route:",
                record["route"],
            ]
        )
    if result.get("raw_final_output") is not None:
        lines.extend(
            [
                "",
                SUBDIVIDER,
                "FINAL ANSWER",
                SUBDIVIDER,
                "Generator status:",
                "Produced candidate JSON",
            ]
        )
    for record in result.get("output_validation_history", []):
        lines.extend(
            [
                "",
                "OUTPUT VALIDATION",
                f"Attempt {record['attempt']} : {record['status']}",
            ]
        )
        if record["error"]:
            error_summary = record["error"].splitlines()[0]
            lines.append(f"Reason : {error_summary}")
        lines.append(f"Route : {record['route']}")
        if record["status"] == "INVALID" and record["route"].endswith("Output Repair"):
            lines.extend(
                [
                    "",
                    "OUTPUT REPAIR",
                    f"Repair attempt : {result.get('output_repair_count', 0)}",
                ]
            )
    lines.extend(
        [
            "",
            DIVIDER,
            "FINAL RESULT",
            DIVIDER,
            f"Status       : {result['status']}",
            f"Replan count : {result['replan_count']}",
            f"Goals completed : {len(result.get('completed_goal_results', []))}",
            f"Trusted-tool calls : {result.get('trusted_tool_calls', 0)}",
            f"Generated scripts : {result.get('generated_script_count', 0)}",
            f"Code repairs : {result.get('code_repair_count', 0)}",
            "",
            "JSON:",
            result["final_answer"],
            "",
            f"Detailed log: {log_path}",
            f"Run artifacts: {result.get('run_directory', log_path.parent)}",
        ]
    )
    return "\n".join(lines)


def build_demo_parser() -> argparse.ArgumentParser:
    """Create the existing workflow-demo command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "live"), required=True)
    parser.add_argument(
        "--scenario",
        choices=(
            "happy",
            "replan",
            "max-replan",
            "valid-json",
            "output-repair",
            "malformed-json",
            "output-failure",
            "trusted-tools-success",
            "generated-python-success",
            "generated-python-repair",
            "generated-python-failure",
        ),
        default="happy",
        help="Scripted scenario used in offline mode",
    )
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--file", dest="file_paths", type=Path, action="append")
    parser.add_argument("--max-replans", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
    return parser


def build_evaluation_parser() -> argparse.ArgumentParser:
    """Create the live Verifier evaluation parser."""
    parser = argparse.ArgumentParser(
        prog="python -m data_analysis_agent.demo verifier-eval",
        description="Run fixed gold cases through only the live Nebius Verifier.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "examples/verifier_cases.json",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
    return parser


def _run_evaluation_command(arguments: list[str]) -> int:
    args = build_evaluation_parser().parse_args(arguments)
    try:
        cases = load_verifier_cases(args.cases)
        settings = load_settings()
        model_id = args.model or settings.nebius_model
        model = NebiusRoleModel(
            client=create_nebius_client(settings),
            model=model_id,
            temperature=0,
        )
        run = run_verifier_evaluation(
            cases=cases,
            cases_path=args.cases,
            model=model,
            model_id=model_id,
            repeats=args.repeats,
            output_dir=args.output_dir,
            project_root=PROJECT_ROOT,
        )
    except (
        ConfigurationError,
        EvaluationFixtureError,
        EvaluationOutputError,
    ) as error:
        print(f"Verifier evaluation failed: {error}", file=sys.stderr)
        return 1
    print(format_evaluation_summary(run))
    return 0 if run.metrics.evaluated_judgments > 0 else 1


def main(argv: list[str] | None = None) -> int:
    """Run a workflow demo or the manual live Verifier evaluation."""
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "verifier-eval":
        return _run_evaluation_command(arguments[1:])

    args = build_demo_parser().parse_args(arguments)
    try:
        log_path = _create_demo_log_path(args.output_dir)
        result = run_demo(
            mode=args.mode,
            scenario=args.scenario,
            prompt_path=args.prompt,
            file_paths=args.file_paths,
            max_replans=args.max_replans,
            log_path=log_path,
        )
    except (
        ConfigurationError,
        DemoInputError,
        OpenAIError,
        RuntimeError,
        VerifierOutputError,
    ) as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        return 1
    print(
        format_workflow_result(
            mode=args.mode,
            scenario=args.scenario,
            result=result,
            log_path=log_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
