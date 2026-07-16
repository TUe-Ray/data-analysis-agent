#!/usr/bin/env python3
"""Export the static LangGraph orchestration diagram as Mermaid and PNG."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from data_analysis_agent.config import load_settings
from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import NebiusRoleModel
from data_analysis_agent.nebius_client import create_nebius_client
from data_analysis_agent.python_runner import LocalPythonRunner


def export_graph(graph: Any, output_dir: Path, *, include_png: bool = True) -> None:
    """Write the graph's Mermaid source and, optionally, a Mermaid PNG render."""
    output_dir.mkdir(parents=True, exist_ok=True)
    drawable = graph.get_graph()
    (output_dir / "agent_workflow.mmd").write_text(
        drawable.draw_mermaid(), encoding="utf-8"
    )
    if include_png:
        (output_dir / "agent_workflow.png").write_bytes(drawable.draw_mermaid_png())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the small command-line interface used by ``make graph-export``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/graphs"))
    parser.add_argument(
        "--mermaid-ink-png",
        action="store_true",
        help=(
            "also write a PNG through Mermaid.Ink; this sends Mermaid source "
            "to that external rendering service"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build the live graph configuration without sending a model request."""
    args = parse_args(argv)
    settings = load_settings()
    model = NebiusRoleModel(
        client=create_nebius_client(settings),
        model=settings.nebius_model,
    )
    workflow = build_graph(model=model, runner=LocalPythonRunner())
    export_graph(workflow, args.output_dir, include_png=args.mermaid_ink_png)
    print(f"Wrote graph files to {args.output_dir}")


if __name__ == "__main__":
    main()
