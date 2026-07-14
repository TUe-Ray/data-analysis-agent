"""Command-line demo for the minimal verification-first LangGraph workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, cast

from openai import OpenAIError

from data_analysis_agent.config import ConfigurationError, load_settings
from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import NebiusRoleModel, RoleModel, build_scripted_model
from data_analysis_agent.nebius_client import create_nebius_client
from data_analysis_agent.nodes import VerifierOutputError
from data_analysis_agent.state import AgentState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    if scenario in {"replan", "max-replan"}:
        return (
            PROJECT_ROOT / "examples/prompts/verifier_trap.txt",
            [PROJECT_ROOT / "examples/data/measurements_with_missing.csv"],
        )
    raise DemoInputError(f"Unknown offline scenario: {scenario}")


def run_demo(
    *,
    mode: Literal["offline", "live"],
    scenario: str = "happy",
    prompt_path: Path | None = None,
    file_paths: list[Path] | None = None,
    max_replans: int = 1,
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
    elif mode == "live":
        if prompt_path is None or not file_paths:
            raise DemoInputError("Live mode requires --prompt and at least one --file")
        settings = load_settings()
        model = NebiusRoleModel(
            client=create_nebius_client(settings),
            model=settings.nebius_model,
        )
    else:
        raise DemoInputError(f"Unknown demo mode: {mode}")

    question = _read_small_text_file(prompt_path).strip()
    staged_names, input_context = stage_input_files(file_paths)
    graph = build_graph(model)
    result = graph.invoke(
        {
            "question": question,
            "file_paths": staged_names,
            "input_context": input_context,
            "replan_count": 0,
            "max_replans": max_replans,
            "trace": [],
        }
    )
    return cast(AgentState, result)


def print_result(*, mode: str, result: AgentState) -> None:
    """Print the concise public state produced by a demo run."""
    print(f"Mode: {mode}")
    print(f"Question: {result['question']}")
    print(f"Staged files: {', '.join(result['file_paths'])}")
    print(f"Plan:\n{result['plan']}")
    print(f"Execution result: {result['execution_result']}")
    print(f"Verification decision: {result['verification_decision']}")
    print(f"Verification feedback: {result['verification_feedback']}")
    print(f"Replan count: {result['replan_count']}")
    print(f"Trace: {' -> '.join(result['trace'])}")
    print(f"Final status: {result['status']}")
    print(f"Final answer: {result['final_answer']}")


def build_parser() -> argparse.ArgumentParser:
    """Create the Prototype V0 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "live"), required=True)
    parser.add_argument(
        "--scenario",
        choices=("happy", "replan", "max-replan"),
        default="happy",
        help="Scripted scenario used in offline mode",
    )
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--file", dest="file_paths", type=Path, action="append")
    parser.add_argument("--max-replans", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""
    args = build_parser().parse_args(argv)
    try:
        result = run_demo(
            mode=args.mode,
            scenario=args.scenario,
            prompt_path=args.prompt,
            file_paths=args.file_paths,
            max_replans=args.max_replans,
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
    print_result(mode=args.mode, result=result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
