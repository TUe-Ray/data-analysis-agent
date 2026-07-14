"""Structured outputs used by the Prototype V0 verifier."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VerificationOutput(BaseModel):
    """Validated routing decision returned by the Verifier."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["PASS", "REPLAN"]
    feedback: str = Field(min_length=1)
