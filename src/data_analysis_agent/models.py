"""Small role-model adapters for scripted and live Nebius execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from openai import OpenAI

Role = Literal["planner", "executor", "verifier"]


class RoleModel(Protocol):
    """Minimal interface shared by offline and live role models."""

    def generate(self, *, role: Role, messages: list[dict[str, str]]) -> str:
        """Generate one response for a role-specific message context."""
        ...


@dataclass(frozen=True)
class ModelCall:
    """One recorded scripted-model call, exposed for deterministic tests."""

    role: Role
    messages: list[dict[str, str]]


class ScriptedRoleModel:
    """Return scenario responses in order without contacting a network."""

    def __init__(self, responses: dict[Role, list[str]]) -> None:
        self._responses = {role: list(values) for role, values in responses.items()}
        self._positions: dict[Role, int] = {
            "planner": 0,
            "executor": 0,
            "verifier": 0,
        }
        self.calls: list[ModelCall] = []

    def generate(self, *, role: Role, messages: list[dict[str, str]]) -> str:
        """Return the next scripted response and record the isolated context."""
        self.calls.append(
            ModelCall(role=role, messages=[dict(message) for message in messages])
        )
        position = self._positions[role]
        responses = self._responses.get(role, [])
        if position >= len(responses):
            raise RuntimeError(f"No scripted {role} response remains")
        self._positions[role] = position + 1
        return responses[position]


class NebiusRoleModel:
    """Use one configured Nebius model for each V0 role."""

    def __init__(self, *, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def generate(self, *, role: Role, messages: list[dict[str, str]]) -> str:
        """Send a role-specific request through the OpenAI-compatible client."""
        del role
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Nebius returned an empty model response")
        return content


def build_scripted_model(scenario: str) -> ScriptedRoleModel:
    """Build one of the deterministic Prototype V0 demo scenarios."""
    complete_plan = """1. Identify non-missing values.
2. Calculate the arithmetic mean.
3. Report the number of observations used."""
    incomplete_plan = """1. Identify non-missing values.
2. Calculate the arithmetic mean.
3. Report the mean."""
    repaired_plan = """1. Identify non-missing values.
2. Calculate the arithmetic mean.
3. Calculate the sample standard error.
4. Report the number of observations used.
5. Report all requested outputs."""
    pass_happy = (
        '{"decision":"PASS","feedback":"The result answers all requested items."}'
    )
    request_replan = (
        '{"decision":"REPLAN","feedback":"The user also requested the sample '
        'standard error and the number of observations used."}'
    )
    pass_recovery = (
        '{"decision":"PASS","feedback":"All requested outputs are now present '
        'and consistent with the supplied data."}'
    )

    if scenario == "happy":
        return ScriptedRoleModel(
            {
                "planner": [complete_plan],
                "executor": ["Mean = 13. Number of observations used = 4."],
                "verifier": [pass_happy],
            }
        )
    if scenario == "replan":
        return ScriptedRoleModel(
            {
                "planner": [incomplete_plan, repaired_plan],
                "executor": [
                    "Mean = 13.",
                    (
                        "Mean = 13. Sample standard error = 1.291. "
                        "Number of observations used = 4."
                    ),
                ],
                "verifier": [request_replan, pass_recovery],
            }
        )
    if scenario == "max-replan":
        return ScriptedRoleModel(
            {
                "planner": [incomplete_plan, repaired_plan],
                "executor": [
                    "Mean = 13.",
                    (
                        "Mean = 13. Sample standard error = 1.291. "
                        "Number of observations used = 4."
                    ),
                ],
                "verifier": [request_replan, request_replan],
            }
        )
    raise ValueError(f"Unknown offline scenario: {scenario}")
