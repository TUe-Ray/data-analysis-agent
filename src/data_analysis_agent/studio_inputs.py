"""Minimal public input contract and adapter used only by LangGraph Studio."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from data_analysis_agent.demo import DemoInputError, stage_input_files
from data_analysis_agent.state import AgentState


class StudioInput(BaseModel):
    """The two fields Studio exposes before an analysis run starts."""

    question: str = Field(
        min_length=1,
        description="The scientific question or analysis instruction.",
    )
    input_data: list[str] = Field(
        min_length=1,
        description=(
            "One or more CSV or UTF-8 text-file paths readable by the local "
            "LangGraph server. Example: ['/absolute/path/measurements.csv']."
        ),
    )


def normalize_studio_input_path(raw_path: str) -> Path:
    """Resolve a local or Windows path into the WSL server's filesystem view."""
    candidate = raw_path.strip().strip('"').strip("'")
    windows_drive_path = re.fullmatch(r"([A-Za-z]):[\\/](.*)", candidate)
    if windows_drive_path:
        drive, remainder = windows_drive_path.groups()
        components = [part for part in re.split(r"[\\/]+", remainder) if part]
        return Path("/mnt") / drive.lower() / Path(*components)
    return Path(candidate).expanduser()


def prepare_studio_input(state: AgentState) -> dict[str, object]:
    """Validate and stage Studio's concise input into the existing AgentState."""
    payload = StudioInput.model_validate(
        {
            "question": state.get("question"),
            "input_data": state.get("input_data"),
        }
    )
    paths = [
        normalize_studio_input_path(raw_path).resolve()
        for raw_path in payload.input_data
    ]
    try:
        file_paths, input_context = stage_input_files(paths)
    except DemoInputError as error:
        raise ValueError(f"Invalid Studio input_data: {error}") from error
    return {
        "question": payload.question,
        "input_data": [str(path) for path in paths],
        "file_paths": file_paths,
        "staged_file_paths": [str(path) for path in paths],
        "input_context": input_context,
    }
