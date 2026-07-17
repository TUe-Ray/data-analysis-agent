"""Minimal public input contract and adapter used only by LangGraph Studio."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from data_analysis_agent.demo import (
    SUPPORTED_SUFFIXES,
    DemoInputError,
    stage_input_files,
)
from data_analysis_agent.state import AgentState

_SKIPPED_DIRECTORY_NAMES = {".git", ".venv", "__pycache__", "private"}
MAX_STUDIO_INPUT_FILES = 100


class StudioInput(BaseModel):
    """The two fields Studio exposes before an analysis run starts."""

    question: str = Field(
        min_length=1,
        description="The scientific question or analysis instruction.",
    )
    input_data: str = Field(
        min_length=1,
        description=(
            "An absolute file or folder path readable by the local LangGraph "
            "server. Folders are searched recursively for CSV, TXT, Markdown, "
            "and JSON files; private and environment folders are skipped. "
            "Example: 'C:/Users/User1/Downloads/study_inputs'."
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


def discover_studio_input_files(path: Path) -> list[Path]:
    """Return allowed readable files from one selected Studio file or folder."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise DemoInputError(
            f"Input path does not exist or is not a file/folder: {path}"
        )
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in SUPPORTED_SUFFIXES
        and not (set(candidate.relative_to(path).parts) & _SKIPPED_DIRECTORY_NAMES)
    )
    if not files:
        extensions = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise DemoInputError(
            f"Input folder contains no supported files ({extensions}): {path}"
        )
    if len(files) > MAX_STUDIO_INPUT_FILES:
        raise DemoInputError(
            f"Input folder has {len(files)} supported files; the Studio limit is "
            f"{MAX_STUDIO_INPUT_FILES}: {path}"
        )
    names = [candidate.name for candidate in files]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DemoInputError(
            "Input folder has duplicate filenames, which Studio cannot stage "
            f"unambiguously: {', '.join(duplicates)}"
        )
    return files


def prepare_studio_input(state: AgentState) -> dict[str, object]:
    """Validate and stage Studio's concise input into the existing AgentState."""
    payload = StudioInput.model_validate(
        {
            "question": state.get("question"),
            "input_data": state.get("input_data"),
        }
    )
    input_path = normalize_studio_input_path(payload.input_data).resolve()
    try:
        paths = discover_studio_input_files(input_path)
        file_paths, input_context = stage_input_files(paths)
    except DemoInputError as error:
        raise ValueError(f"Invalid Studio input_data: {error}") from error
    return {
        "question": payload.question,
        "input_data": str(input_path),
        "file_paths": file_paths,
        "staged_file_paths": [str(path) for path in paths],
        "input_context": input_context,
    }
