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


def _load_public_files(
    public_root: Path, task_config: dict[str, object], task_id: str
) -> tuple[list[str], dict[str, str]]:
    """Load an explicit public manifest or the legacy ``public/data`` tree."""
    declared = task_config.get("public_files")
    if declared is None:
        data_root = public_root / "data"
        paths = sorted(path for path in data_root.rglob("*") if path.is_file())
        names = [str(path.relative_to(data_root)) for path in paths]
    else:
        if (
            not isinstance(declared, list)
            or not declared
            or any(not isinstance(item, str) for item in declared)
        ):
            raise BenchmarkTaskError("public_files must be a non-empty string list")
        names = list(declared)
        if len(names) != len(set(names)):
            raise BenchmarkTaskError("public_files contains duplicate paths")
        paths = []
        root = public_root.resolve()
        for name in names:
            relative = Path(name)
            candidate = (public_root / relative).resolve()
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or root not in candidate.parents
                or not candidate.is_file()
            ):
                raise BenchmarkTaskError(f"Unsafe or missing public file: {name!r}")
            paths.append(candidate)
    if not paths:
        raise BenchmarkTaskError(f"Task {task_id} has no public data files")
    try:
        return names, {
            name: path.read_text(encoding="utf-8")
            for name, path in zip(names, paths, strict=True)
        }
    except (OSError, UnicodeError) as error:
        raise BenchmarkTaskError(
            f"Could not read task public files: {error}"
        ) from error


def _load_prompt_variants(
    public_root: Path, task_config: dict[str, object], requested_variant: str | None
) -> tuple[str, str]:
    """Select a declared prompt safely, preserving legacy task packages."""
    variants = task_config.get("prompt_variants")
    if variants is None:
        if requested_variant not in (None, "default"):
            raise BenchmarkTaskError(
                f"Task prompt variant {requested_variant!r} is unknown; "
                "this legacy task exposes only 'default'."
            )
        variant, relative_path = "default", "prompt.txt"
    else:
        if not isinstance(variants, dict) or not variants:
            raise BenchmarkTaskError("prompt_variants must be a non-empty object")
        default = task_config.get("default_prompt_variant")
        if not isinstance(default, str) or default not in variants:
            raise BenchmarkTaskError(
                "default_prompt_variant must name a declared prompt variant"
            )
        variant = requested_variant or default
        if variant not in variants:
            raise BenchmarkTaskError(
                f"Task prompt variant {variant!r} is unknown; available variants: "
                + ", ".join(sorted(str(name) for name in variants))
            )
        relative_path = variants[variant]
        if not isinstance(relative_path, str):
            raise BenchmarkTaskError(
                f"Prompt path for variant {variant!r} must be a string"
            )
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise BenchmarkTaskError(
            f"Unsafe prompt path for variant {variant!r}: {relative_path!r}"
        )
    prompt_path = (public_root / path).resolve()
    if public_root.resolve() not in prompt_path.parents or not prompt_path.is_file():
        raise BenchmarkTaskError(
            f"Prompt file for variant {variant!r} does not exist inside public/: "
            f"{relative_path!r}"
        )
    try:
        return variant, prompt_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise BenchmarkTaskError(
            f"Could not read prompt variant {variant!r}: {error}"
        ) from error


def load_benchmark_task(
    tasks_root: Path, task_id: str, prompt_variant: str | None = None
) -> LoadedBenchmarkTask:
    """Read public content and retain private paths outside the public view."""
    task_root = (tasks_root / task_id).resolve()
    public_root = task_root / "public"
    private_root = task_root / "private"
    try:
        task_config = json.loads(
            (public_root / "task.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkTaskError(
            f"Could not load public task {task_id}: {error}"
        ) from error
    variant, prompt = _load_prompt_variants(public_root, task_config, prompt_variant)
    data_files, data_contents = _load_public_files(public_root, task_config, task_id)

    grader = private_root / "grader.py"
    reference = private_root / "reference.json"
    if not grader.is_file() or not reference.is_file():
        raise BenchmarkTaskError(f"Task {task_id} is missing private grading files")
    public = PublicTaskView(
        task_id=task_id,
        prompt_variant=variant,
        prompt=prompt,
        data_files=data_files,
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
    """Copy only manifested in-memory public files into a clean attempt."""
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
