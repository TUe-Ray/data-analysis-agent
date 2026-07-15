"""Load benchmark task packages while keeping private files structurally separate."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from data_analysis_agent.benchmark_types import (
    LoadedBenchmarkTask,
    PrivateGradingSpec,
    PublicTaskView,
)


class BenchmarkTaskError(ValueError):
    """Raised when a public/private benchmark task package is malformed."""


def load_benchmark_task(tasks_root: Path, task_id: str) -> LoadedBenchmarkTask:
    """Read public content and retain private paths outside the public view."""
    task_root = (tasks_root / task_id).resolve()
    public_root = task_root / "public"
    private_root = task_root / "private"
    try:
        task_config = json.loads(
            (public_root / "task.json").read_text(encoding="utf-8")
        )
        prompt = (public_root / "prompt.txt").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkTaskError(
            f"Could not load public task {task_id}: {error}"
        ) from error
    data_root = public_root / "data"
    data_paths = sorted(path for path in data_root.rglob("*") if path.is_file())
    if not data_paths:
        raise BenchmarkTaskError(f"Task {task_id} has no public data files")
    data_contents: dict[str, str] = {}
    try:
        for path in data_paths:
            name = str(path.relative_to(data_root))
            data_contents[name] = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise BenchmarkTaskError(f"Could not read task data: {error}") from error

    grader = private_root / "grader.py"
    reference = private_root / "reference.json"
    if not grader.is_file() or not reference.is_file():
        raise BenchmarkTaskError(f"Task {task_id} is missing private grading files")
    public = PublicTaskView(
        task_id=task_id,
        prompt=prompt,
        data_files=list(data_contents),
        data_contents=data_contents,
        answer_schema=task_config["answer_schema"],
        metadata=task_config.get("metadata", {}),
    )
    return LoadedBenchmarkTask(
        public=public,
        private=PrivateGradingSpec(
            grader_path=str(grader),
            reference_path=str(reference),
        ),
    )


def stage_public_task(public: PublicTaskView, destination: Path) -> PublicTaskView:
    """Copy only in-memory public data into a clean approach directory."""
    inputs = destination / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    staged_files: list[str] = []
    staged_contents: dict[str, str] = {}
    for name in public.data_files:
        source_name = Path(name)
        if source_name.is_absolute() or ".." in source_name.parts:
            raise BenchmarkTaskError(f"Unsafe public data filename: {name}")
        target = inputs / source_name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = public.data_contents[name]
        target.write_text(content, encoding="utf-8")
        staged_path = (Path("inputs") / source_name).as_posix()
        staged_files.append(staged_path)
        staged_contents[staged_path] = content
    return public.model_copy(
        update={"data_files": staged_files, "data_contents": staged_contents}
    )


def reset_attempt_directory(path: Path) -> None:
    """Remove prior state; public staging creates the directory when needed."""
    if path.exists():
        shutil.rmtree(path)
