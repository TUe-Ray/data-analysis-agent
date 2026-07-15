"""Public/private task boundaries and persisted benchmark result models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

Approach = Literal["direct_answer", "one_shot_code", "agent", "single_agent_checker"]
AttemptStatus = Literal[
    "completed",
    "wrong_answer",
    "invalid_json",
    "execution_failed",
    "python_policy_failure",
    "timed_out",
    "infrastructure_error",
    "not_applicable",
    "error",
]


class PublicTaskView(BaseModel):
    """The complete and only task representation exposed to model-facing code."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    prompt: str
    data_files: list[str]
    data_contents: dict[str, str]
    answer_schema: dict[str, JsonValue]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PrivateGradingSpec(BaseModel):
    """Private paths available only to the post-execution grading layer."""

    model_config = ConfigDict(extra="forbid")

    grader_path: str
    reference_path: str


class LoadedBenchmarkTask(BaseModel):
    """Orchestrator-only task package with an explicit privacy boundary."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    public: PublicTaskView
    private: PrivateGradingSpec


class GradeResult(BaseModel):
    """Deterministic external grade produced without model involvement."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)
    details: dict[str, JsonValue] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """Shared generation and execution settings for a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = 0.0
    top_p: float | None = None
    max_output_tokens: int = Field(default=4096, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    direct_answer_max_input_chars: int = Field(default=500_000, gt=0)
    max_replans: int = Field(default=1, ge=0)
    repeats: int = Field(default=1, gt=0)
    task_ids: list[str]
    approaches: list[Approach]
    live: bool = False
    live_progress: bool = True
    stop_after_goals: int | None = Field(default=None, gt=0)


class ApproachOutcome(BaseModel):
    """Ungraded candidate and operational facts from one approach."""

    model_config = ConfigDict(extra="forbid")

    status: AttemptStatus
    candidate: dict[str, JsonValue] | None = None
    api_call_count: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    transport_retry_count: int = 0
    execution_exit_code: int | None = None
    timed_out: bool = False
    generated_script_count: int = 0
    local_repair_count: int = 0
    global_replan_count: int = 0
    global_checker_repair_count: int = 0
    run_error: str | None = None
    error_category: str | None = None
    exception_class: str | None = None
    not_applicable_reason: str | None = None
    verifier_decisions: list[str] = Field(default_factory=list)
    partial_run: bool = False
    partial_run_reached: bool = False
    partial_goal_id: str | None = None


class BenchmarkResult(BaseModel):
    """One persisted row per task, approach, and repeat."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    approach: Approach
    repeat_index: int
    model: str
    status: AttemptStatus
    graded: bool
    graded_success: bool
    grader_score: float | None
    grader_errors: list[str]
    grader_details: dict[str, JsonValue]
    api_call_count: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    transport_retry_count: int
    wall_clock_latency: float
    execution_exit_code: int | None
    timed_out: bool
    generated_script_count: int
    local_repair_count: int
    global_replan_count: int
    global_checker_repair_count: int = 0
    final_candidate_json: dict[str, JsonValue] | None
    artifact_directory: str
    run_error: str | None = None
    error_category: str | None = None
    exception_class: str | None = None
    not_applicable_reason: str | None = None
    verifier_decisions: list[str] = Field(default_factory=list)
    partial_run: bool = False
    partial_run_reached: bool = False
    partial_goal_id: str | None = None


class ApproachMetrics(BaseModel):
    """Aggregate metrics for one approach."""

    model_config = ConfigDict(extra="forbid")

    attempted_runs: int
    passed_runs: int
    pass_rate: float
    invalid_json_count: int
    code_execution_failure_count: int
    timeout_count: int
    average_api_calls: float
    average_total_tokens: float | None
    average_latency: float
    average_generated_script_versions: float
    average_local_repair_count: float
    average_global_replan_count: float
    average_global_checker_repair_count: float = 0.0


class BenchmarkSummary(BaseModel):
    """Detailed result-file summary for one benchmark invocation."""

    model_config = ConfigDict(extra="forbid")

    benchmark_run_id: str
    config: BenchmarkConfig
    metrics: dict[str, ApproachMetrics]
    results_path: str


def relative_to(path: Path, root: Path) -> str:
    """Prefer a stable repository-relative artifact path."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())
