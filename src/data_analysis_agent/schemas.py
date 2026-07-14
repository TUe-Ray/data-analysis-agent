"""Structured verifier and final-answer outputs for Prototype V0."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter


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
