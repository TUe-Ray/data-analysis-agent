"""Structured workflow, execution, verifier, and final-answer outputs."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter


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
