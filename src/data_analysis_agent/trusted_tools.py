"""Small deterministic trusted-tool registry for staged text and CSV files."""

from __future__ import annotations

import csv
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from data_analysis_agent.schemas import ToolExecutionResult


class InspectFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str


class InspectFileOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_name: str
    format: Literal["csv", "text"]
    size_bytes: int
    readable: bool
    row_count: int | None = None
    column_names: list[str] | None = None
    warnings: list[str]


class ProfileTableInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str
    selected_columns: list[str] | None = None
    sample_rows: int = Field(default=5, ge=0, le=20)


class ProfileTableOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row_count: int
    column_count: int
    column_names: list[str]
    inferred_dtypes: dict[str, str]
    missing_count: dict[str, int]
    unique_count: dict[str, int]
    duplicate_row_count: int
    numeric_ranges: dict[str, dict[str, float]]
    preview: list[dict[str, JsonValue]]
    warnings: list[str]


StatisticName = Literal[
    "count",
    "mean",
    "median",
    "sample_standard_deviation",
    "sample_standard_error",
    "minimum",
    "maximum",
    "quantiles",
]


class SummaryStatisticsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str
    column: str
    statistics: list[StatisticName] = Field(min_length=1)
    drop_missing: bool
    quantiles: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75])


class SummaryStatisticsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str
    column: str
    statistics: dict[str, JsonValue]
    input_row_count: int
    rows_used: int
    missing_rows_excluded: int
    warnings: list[str]


@dataclass(frozen=True)
class ToolDefinition:
    """Minimal validated definition for one trusted local capability."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    limitations: str
    handler: Callable[[BaseModel], BaseModel]


def _is_missing(value: str | None) -> bool:
    return value is None or value.strip() == ""


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file does not contain a header row")
        fields = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return fields, rows


def _infer_dtype(values: list[str]) -> str:
    usable = [value for value in values if not _is_missing(value)]
    if not usable:
        return "unknown"
    try:
        for value in usable:
            int(value)
    except ValueError:
        try:
            for value in usable:
                float(value)
        except ValueError:
            return "string"
        return "float"
    return "integer"


def _json_value(value: str) -> JsonValue:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _quantile(values: list[float], probability: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError(f"Quantile must be between 0 and 1: {probability}")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class TrustedToolRegistry:
    """Registry whose handlers can access only explicitly staged local files."""

    def __init__(
        self,
        *,
        allowed_roots: list[Path],
        allowed_files: list[Path] | None = None,
    ) -> None:
        self.allowed_roots = [path.resolve() for path in allowed_roots]
        self.allowed_files = (
            {path.resolve() for path in allowed_files}
            if allowed_files is not None
            else None
        )
        self._tools = self._build_tools()

    @property
    def names(self) -> set[str]:
        return set(self._tools)

    def _resolve_file(self, supplied: str) -> Path:
        raw = Path(supplied)
        candidates = [raw.resolve()] if raw.is_absolute() else []
        candidates.extend((root / raw).resolve() for root in self.allowed_roots)
        if self.allowed_files is not None and len(raw.parts) == 1:
            candidates.extend(
                path for path in self.allowed_files if path.name == raw.name
            )
        for candidate in candidates:
            within_root = any(
                candidate == root or root in candidate.parents
                for root in self.allowed_roots
            )
            explicitly_staged = (
                self.allowed_files is None or candidate in self.allowed_files
            )
            if within_root and explicitly_staged and candidate.is_file():
                return candidate
        raise ValueError("file_path is not an explicitly staged local file")

    def _build_tools(self) -> dict[str, ToolDefinition]:
        def inspect_file(value: BaseModel) -> BaseModel:
            request = InspectFileInput.model_validate(value)
            path = self._resolve_file(request.file_path)
            suffix = path.suffix.lower()
            if suffix not in {".csv", ".txt"}:
                raise ValueError("inspect_file supports only CSV and plain text")
            row_count = None
            columns = None
            if suffix == ".csv":
                columns, rows = _load_csv(path)
                row_count = len(rows)
            else:
                path.read_text(encoding="utf-8")
            return InspectFileOutput(
                file_name=path.name,
                format="csv" if suffix == ".csv" else "text",
                size_bytes=path.stat().st_size,
                readable=True,
                row_count=row_count,
                column_names=columns,
                warnings=[],
            )

        def profile_table(value: BaseModel) -> BaseModel:
            request = ProfileTableInput.model_validate(value)
            path = self._resolve_file(request.file_path)
            if path.suffix.lower() != ".csv":
                raise ValueError("profile_table supports CSV files only")
            fields, rows = _load_csv(path)
            selected = request.selected_columns or fields
            unknown = [column for column in selected if column not in fields]
            if unknown:
                raise ValueError(f"Unknown selected columns: {', '.join(unknown)}")
            dtypes: dict[str, str] = {}
            missing: dict[str, int] = {}
            unique: dict[str, int] = {}
            ranges: dict[str, dict[str, float]] = {}
            for column in selected:
                values = [row.get(column, "") for row in rows]
                dtype = _infer_dtype(values)
                dtypes[column] = dtype
                missing[column] = sum(_is_missing(item) for item in values)
                unique[column] = len({item for item in values if not _is_missing(item)})
                if dtype in {"integer", "float"}:
                    numeric = [float(item) for item in values if not _is_missing(item)]
                    if numeric:
                        ranges[column] = {
                            "minimum": min(numeric),
                            "maximum": max(numeric),
                        }
            row_keys = [tuple(row.get(field, "") for field in fields) for row in rows]
            duplicate_count = len(row_keys) - len(set(row_keys))
            preview = [
                {column: _json_value(row.get(column, "")) for column in selected}
                for row in rows[: request.sample_rows]
            ]
            return ProfileTableOutput(
                row_count=len(rows),
                column_count=len(fields),
                column_names=fields,
                inferred_dtypes=dtypes,
                missing_count=missing,
                unique_count=unique,
                duplicate_row_count=duplicate_count,
                numeric_ranges=ranges,
                preview=preview,
                warnings=[],
            )

        def compute_summary_statistics(value: BaseModel) -> BaseModel:
            request = SummaryStatisticsInput.model_validate(value)
            path = self._resolve_file(request.file_path)
            if path.suffix.lower() != ".csv":
                raise ValueError("compute_summary_statistics supports CSV files only")
            fields, rows = _load_csv(path)
            if request.column not in fields:
                raise ValueError(f"Unknown column: {request.column}")
            raw_values = [row.get(request.column, "") for row in rows]
            missing_count = sum(_is_missing(item) for item in raw_values)
            if missing_count and not request.drop_missing:
                raise ValueError(
                    "Missing values are present; set drop_missing=true or revise "
                    "the goal"
                )
            try:
                values = [float(item) for item in raw_values if not _is_missing(item)]
            except ValueError as error:
                raise ValueError(f"Column {request.column!r} is not numeric") from error
            if any(not math.isfinite(item) for item in values):
                raise ValueError("Numeric column contains a non-finite value")
            if not values:
                raise ValueError("No usable numeric observations remain")
            requested = list(dict.fromkeys(request.statistics))
            if (
                any(
                    name in requested
                    for name in ("sample_standard_deviation", "sample_standard_error")
                )
                and len(values) < 2
            ):
                raise ValueError(
                    "At least two usable values are required for sample deviation "
                    "or error"
                )
            results: dict[str, JsonValue] = {}
            for name in requested:
                if name == "count":
                    results[name] = len(values)
                elif name == "mean":
                    results[name] = statistics.fmean(values)
                elif name == "median":
                    results[name] = statistics.median(values)
                elif name == "sample_standard_deviation":
                    results[name] = statistics.stdev(values)
                elif name == "sample_standard_error":
                    results[name] = statistics.stdev(values) / math.sqrt(len(values))
                elif name == "minimum":
                    results[name] = min(values)
                elif name == "maximum":
                    results[name] = max(values)
                elif name == "quantiles":
                    results[name] = {
                        str(probability): _quantile(values, probability)
                        for probability in request.quantiles
                    }
            return SummaryStatisticsOutput(
                file_path=request.file_path,
                column=request.column,
                statistics=results,
                input_row_count=len(rows),
                rows_used=len(values),
                missing_rows_excluded=missing_count,
                warnings=[],
            )

        return {
            "inspect_file": ToolDefinition(
                name="inspect_file",
                description=(
                    "Inspect one staged CSV or plain-text file without modifying it."
                ),
                input_model=InspectFileInput,
                output_model=InspectFileOutput,
                limitations=(
                    "Local staged files only; no network, directories, or binary "
                    "scientific formats."
                ),
                handler=inspect_file,
            ),
            "profile_table": ToolDefinition(
                name="profile_table",
                description=(
                    "Produce a deterministic basic profile of a staged CSV table."
                ),
                input_model=ProfileTableInput,
                output_model=ProfileTableOutput,
                limitations=(
                    "CSV and basic profiling only; does not infer scientific meaning "
                    "or select statistical methods."
                ),
                handler=profile_table,
            ),
            "compute_summary_statistics": ToolDefinition(
                name="compute_summary_statistics",
                description=(
                    "Compute deterministic descriptive statistics for one numeric "
                    "CSV column."
                ),
                input_model=SummaryStatisticsInput,
                output_model=SummaryStatisticsOutput,
                limitations=(
                    "One numeric column; no repeated-measure correction, causal or "
                    "inferential interpretation, or automatic method selection."
                ),
                handler=compute_summary_statistics,
            ),
        }

    def catalog(self) -> list[dict[str, JsonValue]]:
        """Return the intentionally short capability catalog shown to Executor."""
        return [
            {
                "name": tool.name,
                "purpose": tool.description,
                "inputs": list(
                    tool.input_model.model_json_schema().get("properties", {})
                ),
                "outputs": list(
                    tool.output_model.model_json_schema().get("properties", {})
                ),
                "limitations": tool.limitations,
            }
            for tool in self._tools.values()
        ]

    def execute(
        self, name: str, arguments: dict[str, JsonValue]
    ) -> ToolExecutionResult:
        """Validate arguments and output around one deterministic handler call."""
        started = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return ToolExecutionResult(
                success=False,
                tool_name=name,
                output={},
                warnings=[],
                error=f"Unknown trusted tool: {name}",
                duration_seconds=time.perf_counter() - started,
            )
        try:
            validated_input = tool.input_model.model_validate(arguments)
            raw_output = tool.handler(validated_input)
            output = tool.output_model.model_validate(raw_output)
            output_data = output.model_dump(mode="json")
            warnings = list(output_data.get("warnings", []))
            return ToolExecutionResult(
                success=True,
                tool_name=name,
                output=output_data,
                warnings=warnings,
                duration_seconds=time.perf_counter() - started,
            )
        except (OSError, UnicodeError, ValueError, ValidationError, csv.Error) as error:
            return ToolExecutionResult(
                success=False,
                tool_name=name,
                output={},
                warnings=[],
                error=f"{type(error).__name__}: {error}",
                duration_seconds=time.perf_counter() - started,
            )
