"""Structured workflow, execution, verifier, and final-answer outputs."""

from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
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
    code: str = Field(min_length=1)
    summary: str

    @field_validator("code")
    @classmethod
    def code_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only source even though it satisfies min_length."""
        if not value.strip():
            raise ValueError("code must contain non-whitespace Python source")
        return value


class PythonRepair(BaseModel):
    """Strict structured response for a mechanical Python repair."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["python_repair"]
    code: str = Field(min_length=1)
    summary: str
    addressed_failure_category: ExecutionFailureCategory

    @field_validator("code")
    @classmethod
    def code_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only repaired source."""
        if not value.strip():
            raise ValueError("code must contain non-whitespace Python source")
        return value


class IntermediateGoal(BaseModel):
    """One implementation-agnostic scientific objective in an ordered plan."""

    model_config = ConfigDict(extra="forbid")

    goal_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    required_outputs: list[str]
    constraints: list[str]
    success_criteria: list[str]
    depends_on: list[str] = Field(default_factory=list)


class HighLevelPlan(BaseModel):
    """The Planner's global scientific objective and ordered goals."""

    model_config = ConfigDict(extra="forbid")

    scientific_objective: str = Field(min_length=1)
    goals: list[IntermediateGoal] = Field(min_length=1)


class ExecutionStrategy(BaseModel):
    """The Executor's concise local capability decision for one goal."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["trusted_tool", "generated_python"]
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
    strategy: Literal["trusted_tool", "generated_python"]
    capability_name: str | None = None
    result: dict[str, JsonValue]
    warnings: list[str]
    error: str | None = None
    artifact_paths: list[str]


class VerificationOutput(BaseModel):
    """Validated routing decision returned by the Verifier."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["PASS", "REPLAN"]
    feedback: str = Field(min_length=1)


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
