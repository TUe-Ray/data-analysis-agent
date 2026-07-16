"""Shared public-only factual context for benchmark analysis architectures."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from pydantic import JsonValue

from data_analysis_agent.benchmark_types import PublicTaskView


def _public_name(staged_name: str) -> str:
    path = Path(staged_name)
    if path.parts and path.parts[0] == "inputs":
        return Path(*path.parts[1:]).as_posix()
    return path.as_posix()


def _content(public: PublicTaskView, staged_name: str) -> str:
    if staged_name in public.data_contents:
        return public.data_contents[staged_name]
    public_name = _public_name(staged_name)
    if public_name in public.data_contents:
        return public.data_contents[public_name]
    matches = [
        content
        for name, content in public.data_contents.items()
        if Path(name).name == Path(staged_name).name
    ]
    if len(matches) != 1:
        raise ValueError(f"Could not map public file content for {staged_name!r}")
    return matches[0]


def _scalar_type(values: list[str]) -> str:
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


def _document_names(public: PublicTaskView) -> set[str]:
    declared = public.metadata.get("document_files", [])
    if not isinstance(declared, list):
        return set()
    return {str(item) for item in declared if isinstance(item, str)}


def build_public_analysis_context(
    public: PublicTaskView,
    staged_files: list[Path],
) -> dict[str, JsonValue]:
    """Build one bounded, JSON-safe factual context used before code generation."""
    if len(staged_files) != len(public.data_files):
        raise ValueError("staged file list does not match the public task manifest")
    documents = _document_names(public)
    specification_documents: list[dict[str, JsonValue]] = []
    csv_profiles: list[dict[str, JsonValue]] = []
    other_files: list[dict[str, JsonValue]] = []
    for supplied, resolved in zip(public.data_files, staged_files, strict=True):
        public_name = _public_name(supplied)
        content = _content(public, supplied)
        resolved_path = str(resolved.resolve())
        if public_name in documents:
            specification_documents.append(
                {
                    "public_relative_filename": public_name,
                    "staged_path": resolved_path,
                    "content": content,
                }
            )
        if Path(public_name).suffix.casefold() == ".csv":
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            columns = list(reader.fieldnames or [])
            csv_profiles.append(
                {
                    "public_relative_filename": public_name,
                    "staged_path": resolved_path,
                    "row_count": len(rows),
                    "columns": [
                        {
                            "name": column,
                            "inferred_scalar_type": _scalar_type(
                                [str(row.get(column, "")) for row in rows]
                            ),
                            "missing_count": sum(
                                not str(row.get(column, "")).strip() for row in rows
                            ),
                        }
                        for column in columns
                    ],
                    "representative_rows": [
                        {column: row.get(column, "") for column in columns}
                        for row in rows[:3]
                    ],
                }
            )
        elif public_name not in documents:
            other_files.append(
                {
                    "public_relative_filename": public_name,
                    "staged_path": resolved_path,
                    "character_count": len(content),
                    "preview": content[:500],
                }
            )
    precedence = public.metadata.get("document_precedence", [])
    return {
        "task": {
            "task_id": public.task_id,
            "prompt_variant": public.prompt_variant,
            "main_public_prompt": public.prompt,
            "public_answer_schema": public.answer_schema,
            "document_precedence": precedence,
        },
        "specification_documents": specification_documents,
        "csv_profiles": csv_profiles,
        "other_public_files": other_files,
    }
