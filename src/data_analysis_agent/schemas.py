"""Structured workflow, execution, verifier, and final-answer outputs."""

from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)

ExecutionFailureCategory: TypeAlias = Literal[
    "policy_error",
    "syntax_error",
    "runtime_error",
    "timeout",
    "result_contract_error",
    "generation_contract_error",
]


class PythonGeneration(BaseModel):
    """Strict structured response used to create one generated script."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["python"]
    code_lines: list[StrictStr] = Field(min_length=1)
    summary: str

    @field_validator("code_lines")
    @classmethod
    def code_lines_are_physical_source_lines(cls, value: list[str]) -> list[str]:
        """Keep the generated source's physical-line structure explicit."""
        if any("\n" in line or "\r" in line for line in value):
            raise ValueError(
                "each code_lines item must contain exactly one physical line"
            )
        if not "\n".join(value).strip():
            raise ValueError("code_lines must contain non-whitespace Python source")
        return value

    def source(self) -> str:
        """Reconstruct the only executable representation of this contract."""
        return "\n".join(self.code_lines) + "\n"


class PythonRepair(BaseModel):
    """Strict structured response for a mechanical Python repair."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["python_repair"]
    code_lines: list[StrictStr] = Field(min_length=1)
    summary: str
    addressed_failure_category: ExecutionFailureCategory

    @field_validator("code_lines")
    @classmethod
    def code_lines_are_physical_source_lines(cls, value: list[str]) -> list[str]:
        """Keep repaired source line-oriented as well."""
        if any("\n" in line or "\r" in line for line in value):
            raise ValueError(
                "each code_lines item must contain exactly one physical line"
            )
        if not "\n".join(value).strip():
            raise ValueError("code_lines must contain non-whitespace Python source")
        return value

    def source(self) -> str:
        """Reconstruct the only executable representation of this contract."""
        return "\n".join(self.code_lines) + "\n"


class GoalArtifact(BaseModel):
    """A verifier-approved analysis output safe for a dependent goal to read."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)
    producer_goal_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    relative_name: str = Field(min_length=1)
    media_type: str | None = None
    description: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)
    columns: list[str] | None = None
    row_count: int | None = Field(default=None, ge=0)


class GoalArtifactDeclaration(BaseModel):
    """A generated script's explicit request to publish one analysis output."""

    model_config = ConfigDict(extra="forbid")

    relative_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    media_type: str | None = None


class IntermediateGoal(BaseModel):
    """One implementation-agnostic scientific objective in an ordered plan."""

    model_config = ConfigDict(extra="forbid")

    goal_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    required_outputs: list[str]
    constraints: list[str]
    success_criteria: list[str]
    depends_on: list[str] = Field(default_factory=list)
    # Machine-readable schema coverage.  Unlike required_outputs this is never
    # inferred from prose and is therefore safe to validate deterministically.
    output_paths: list[str] = Field(default_factory=list)


class HighLevelPlan(BaseModel):
    """The Planner's global scientific objective and ordered goals."""

    model_config = ConfigDict(extra="forbid")

    scientific_objective: str = Field(min_length=1)
    goals: list[IntermediateGoal] = Field(min_length=1)
    final_output_goal_id: str | None = None
    invalidate_from_goal_id: str | None = None


class SuffixReplan(BaseModel):
    """A scientific replan replaces only the unfinished suffix of a plan."""

    model_config = ConfigDict(extra="forbid")

    replace_from_goal_id: str = Field(min_length=1)
    replacement_goals: list[IntermediateGoal] = Field(min_length=1)
    final_output_goal_id: str | None = None
    reason: str = Field(min_length=1)


class ExecutionStrategy(BaseModel):
    """The Executor's concise local capability decision for one goal."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["trusted_tool", "generated_python", "structured_result"]
    capability_name: str | None = None
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    concise_reason: str = Field(min_length=1)


class ToolExecutionResult(BaseModel):
    """Validated envelope returned for every trusted-tool invocation."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    tool_name: str
    output: dict[str, JsonValue]
    warnings: list[str]
    error: str | None = None
    duration_seconds: float | None = None


class GoalResult(BaseModel):
    """Factual result retained after executing one intermediate goal."""

    model_config = ConfigDict(extra="forbid")

    goal_id: str
    success: bool
    strategy: Literal["trusted_tool", "generated_python", "structured_result"]
    capability_name: str | None = None
    result: dict[str, JsonValue]
    warnings: list[str]
    error: str | None = None
    artifact_paths: list[str]


class StructuredResult(BaseModel):
    """Compact non-code execution result for document-reconciliation goals."""

    model_config = ConfigDict(extra="forbid")

    result: dict[str, JsonValue]
    warnings: list[StrictStr] = Field(default_factory=list)


class VerificationOutput(BaseModel):
    """Validated routing decision returned by the Verifier."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["PASS", "RETRY_GOAL", "REPLAN"]
    issue_classification: Literal[
        "none",
        "implementation",
        "result",
        "artifact_handoff",
        "dependency_contract",
        "plan_contract",
        "evidence",
    ] = "none"
    feedback: str = Field(min_length=1)


class FinalCheckerOutput(BaseModel):
    """Independent global completeness decision for the ablation approach."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["PASS", "REPAIR"]
    repair_scope: Literal["none", "format_only", "rerun_analysis"]
    feedback: str = Field(min_length=1)

    @model_validator(mode="after")
    def repair_scope_matches_decision(self) -> "FinalCheckerOutput":
        if self.decision == "PASS" and self.repair_scope != "none":
            raise ValueError("PASS requires repair_scope='none'")
        if self.decision == "REPAIR" and self.repair_scope == "none":
            raise ValueError("REPAIR requires format_only or rerun_analysis scope")
        return self


class FinalAnswer(BaseModel):
    """JSON-only successful answer produced after scientific verification."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "completed_with_limitations"]
    answer: str
    key_results: dict[str, JsonValue]
    limitations: list[str]


class FinalFailureAnswer(BaseModel):
    """Explicit machine-readable result for a workflow that did not succeed."""

    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "stopped_after_max_replans",
        "output_validation_failed",
        "python_policy_failure",
        "mechanical_execution_failed",
        "planner_output_failed",
        "verifier_output_failed",
        "goal_retry_exhausted",
    ]
    answer: str | None
    key_results: dict[str, JsonValue]
    limitations: list[str]
    error: str


FinalWorkflowAnswer = Annotated[
    FinalAnswer | FinalFailureAnswer,
    Field(discriminator="status"),
]
FINAL_WORKFLOW_ANSWER_ADAPTER = TypeAdapter(FinalWorkflowAnswer)
