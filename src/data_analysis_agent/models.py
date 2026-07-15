"""Small role-model adapters for scripted and live Nebius execution."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from openai import OpenAI

Role = Literal[
    "planner",
    "executor",
    "verifier",
    "direct_answer",
    "one_shot_code",
]


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


@dataclass(frozen=True)
class ModelExchange:
    """One complete model exchange retained by the demo logging layer."""

    role: Role
    messages: list[dict[str, str]]
    response: str | None
    latency_seconds: float
    error: str | None = None
    token_usage: dict[str, int] | None = None


class ScriptedRoleModel:
    """Return scenario responses in order without contacting a network."""

    def __init__(self, responses: dict[Role, list[str]]) -> None:
        self._responses = {role: list(values) for role, values in responses.items()}
        self._positions: dict[Role, int] = {
            "planner": 0,
            "executor": 0,
            "verifier": 0,
            "direct_answer": 0,
            "one_shot_code": 0,
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

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._max_output_tokens = max_output_tokens
        self.last_token_usage: dict[str, int] | None = None

    def generate(self, *, role: Role, messages: list[dict[str, str]]) -> str:
        """Send a role-specific request through the OpenAI-compatible client."""
        del role
        self.last_token_usage = None
        arguments: dict[str, object] = {
            "model": self._model,
            "messages": messages,
        }
        if self._temperature is not None:
            arguments["temperature"] = self._temperature
        if self._top_p is not None:
            arguments["top_p"] = self._top_p
        if self._max_output_tokens is not None:
            arguments["max_tokens"] = self._max_output_tokens
        response = self._client.chat.completions.create(**arguments)
        if response.usage is not None:
            self.last_token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Nebius returned an empty model response")
        return content


class RecordingRoleModel:
    """Record exact role messages and raw responses for local run logs."""

    def __init__(self, model: RoleModel) -> None:
        self._model = model
        self.exchanges: list[ModelExchange] = []

    def generate(self, *, role: Role, messages: list[dict[str, str]]) -> str:
        """Delegate generation and retain a secret-free exchange record."""
        started = time.perf_counter()
        copied_messages = [dict(message) for message in messages]
        try:
            response = self._model.generate(role=role, messages=messages)
        except Exception as error:
            self.exchanges.append(
                ModelExchange(
                    role=role,
                    messages=copied_messages,
                    response=None,
                    latency_seconds=time.perf_counter() - started,
                    error=f"{type(error).__name__}: {error}",
                    token_usage=_token_usage(self._model),
                )
            )
            raise
        self.exchanges.append(
            ModelExchange(
                role=role,
                messages=copied_messages,
                response=response,
                latency_seconds=time.perf_counter() - started,
                token_usage=_token_usage(self._model),
            )
        )
        return response


def _token_usage(model: RoleModel) -> dict[str, int] | None:
    """Snapshot optional provider usage without requiring it from fake models."""
    usage = getattr(model, "last_token_usage", None)
    return dict(usage) if isinstance(usage, dict) else None


def build_scripted_model(scenario: str) -> ScriptedRoleModel:
    """Build one of the deterministic Prototype V0 demo scenarios."""
    structured_statistics_plan = json.dumps(
        {
            "scientific_objective": (
                "Compute the requested descriptive statistics without imputing "
                "missing values."
            ),
            "goals": [
                {
                    "goal_id": "understand_input",
                    "objective": (
                        "Inspect the supplied table and identify usable numeric "
                        "observations and missing values."
                    ),
                    "required_outputs": [
                        "table shape",
                        "value-column type",
                        "missing-value count",
                    ],
                    "constraints": ["Do not modify the input file."],
                    "success_criteria": [
                        "The value column is identified.",
                        "Missing values are reported.",
                    ],
                    "depends_on": [],
                },
                {
                    "goal_id": "compute_statistics",
                    "objective": (
                        "Compute the arithmetic mean, sample standard error, and "
                        "number of non-missing observations."
                    ),
                    "required_outputs": [
                        "mean",
                        "sample_standard_error",
                        "n_observations",
                    ],
                    "constraints": [
                        "Do not impute missing values.",
                        "Use sample standard deviation with denominator n - 1.",
                    ],
                    "success_criteria": [
                        "All requested values are present and finite."
                    ],
                    "depends_on": ["understand_input"],
                },
            ],
        }
    )
    structured_python_plan = json.dumps(
        {
            "scientific_objective": (
                "Compute the requested sequence statistic from non-missing "
                "measurements."
            ),
            "goals": [
                {
                    "goal_id": "compute_successive_difference",
                    "objective": (
                        "Compute the mean absolute successive difference of the "
                        "non-missing value sequence."
                    ),
                    "required_outputs": ["mean_absolute_successive_difference"],
                    "constraints": [
                        "Preserve input order.",
                        "Do not impute missing values.",
                    ],
                    "success_criteria": ["The requested finite statistic is reported."],
                    "depends_on": [],
                }
            ],
        }
    )
    profile_strategy = json.dumps(
        {
            "strategy": "trusted_tool",
            "capability_name": "profile_table",
            "arguments": {
                "file_path": "measurements_with_missing.csv",
                "sample_rows": 5,
            },
            "concise_reason": (
                "The trusted profiler directly reports table structure and missingness."
            ),
        }
    )
    statistics_strategy = json.dumps(
        {
            "strategy": "trusted_tool",
            "capability_name": "compute_summary_statistics",
            "arguments": {
                "file_path": "measurements_with_missing.csv",
                "column": "value",
                "statistics": ["count", "mean", "sample_standard_error"],
                "drop_missing": True,
            },
            "concise_reason": (
                "The trusted summary tool directly supports all requested calculations."
            ),
        }
    )
    python_strategy = json.dumps(
        {
            "strategy": "generated_python",
            "capability_name": None,
            "arguments": {},
            "concise_reason": "No trusted tool computes successive differences.",
        }
    )
    data_path = (
        Path(__file__).resolve().parents[2]
        / "examples/data/measurements_with_missing.csv"
    )
    good_code = f"""import csv
import json

with open({str(data_path)!r}, encoding="utf-8") as handle:
    values = [float(row["value"]) for row in csv.DictReader(handle) if row["value"]]
differences = [abs(right - left) for left, right in zip(values, values[1:])]
result = sum(differences) / len(differences)
print(json.dumps({{"mean_absolute_successive_difference": result}}))
"""
    wrong_column_code = f"""import csv
import json

with open({str(data_path)!r}, encoding="utf-8") as handle:
    values = [float(row["wrong_value"]) for row in csv.DictReader(handle)]
print(json.dumps({{"mean_absolute_successive_difference": values[0]}}))
"""
    other_wrong_code = wrong_column_code.replace("wrong_value", "still_wrong")
    pass_profile = (
        '{"decision":"PASS","feedback":"The table structure, numeric value '
        'column, and missingness are supported by the profile output."}'
    )
    pass_statistics = (
        '{"decision":"PASS","feedback":"All requested statistics are present '
        'and supported by the trusted tool output."}'
    )
    pass_python = (
        '{"decision":"PASS","feedback":"The successful script produced the '
        'requested sequence statistic without imputation."}'
    )
    fail_python = (
        '{"decision":"REPLAN","feedback":"Generated Python failed and did not '
        'produce the required output."}'
    )

    if scenario == "trusted-tools-success":
        return ScriptedRoleModel(
            {
                "planner": [structured_statistics_plan],
                "executor": [profile_strategy, statistics_strategy],
                "verifier": [pass_profile, pass_statistics],
            }
        )
    if scenario == "generated-python-success":
        return ScriptedRoleModel(
            {
                "planner": [structured_python_plan],
                "executor": [python_strategy, good_code],
                "verifier": [pass_python],
            }
        )
    if scenario == "generated-python-repair":
        return ScriptedRoleModel(
            {
                "planner": [structured_python_plan],
                "executor": [python_strategy, wrong_column_code, good_code],
                "verifier": [pass_python],
            }
        )
    if scenario == "generated-python-failure":
        return ScriptedRoleModel(
            {
                "planner": [structured_python_plan, structured_python_plan],
                "executor": [
                    python_strategy,
                    wrong_column_code,
                    other_wrong_code,
                    python_strategy,
                    wrong_column_code,
                    other_wrong_code,
                ],
                "verifier": [fail_python, fail_python],
            }
        )

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

    if scenario in {
        "valid-json",
        "output-repair",
        "malformed-json",
        "output-failure",
    }:
        return ScriptedRoleModel(
            {
                "planner": [repaired_plan],
                "executor": [
                    (
                        "Mean = 13. Sample standard error = 1.291. "
                        "Number of observations used = 4."
                    )
                ],
                "verifier": [pass_recovery],
            }
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
