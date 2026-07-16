"""Environment-based configuration for Nebius Token Factory."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MAX_CODE_REPAIR_ATTEMPTS = 8
DEFAULT_CODE_REPAIR_NO_PROGRESS_ATTEMPTS = 3
DEFAULT_MAX_PLANNER_REPAIR_ATTEMPTS = 3
DEFAULT_MAX_PLAN_GOALS = 6
DEFAULT_MAX_GOAL_RESULT_BYTES = 262_144
DEFAULT_MAX_GOAL_RESULT_LIST_LENGTH = 100
DEFAULT_MAX_GOAL_RESULT_DEPTH = 10
DEFAULT_MAX_FAILURE_FAMILY_ATTEMPTS = 5
DEFAULT_MAX_MODEL_CALLS = 80
DEFAULT_MAX_GOAL_ROLLBACKS = 1
DEFAULT_MAX_ROLLBACK_GOALS = 6


class ConfigurationError(ValueError):
    """Raised when required Nebius configuration is missing."""


@dataclass(frozen=True)
class Settings:
    """Configuration required to construct a Nebius API client."""

    nebius_api_key: str
    nebius_base_url: str
    nebius_model: str
    max_code_repair_attempts: int = DEFAULT_MAX_CODE_REPAIR_ATTEMPTS
    code_repair_no_progress_attempts: int = DEFAULT_CODE_REPAIR_NO_PROGRESS_ATTEMPTS
    max_planner_repair_attempts: int = DEFAULT_MAX_PLANNER_REPAIR_ATTEMPTS


def _positive_environment_int(name: str, default: int) -> int:
    """Read a positive integer setting with a clear configuration error."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def code_repair_settings() -> tuple[int, int]:
    """Return mechanical-repair limits, including offline graph invocations."""
    return (
        _positive_environment_int(
            "MAX_CODE_REPAIR_ATTEMPTS", DEFAULT_MAX_CODE_REPAIR_ATTEMPTS
        ),
        _positive_environment_int(
            "CODE_REPAIR_NO_PROGRESS_ATTEMPTS",
            DEFAULT_CODE_REPAIR_NO_PROGRESS_ATTEMPTS,
        ),
    )


def max_planner_repair_attempts() -> int:
    """Return the bounded structural Planner-output repair limit."""
    return _positive_environment_int(
        "MAX_PLANNER_REPAIR_ATTEMPTS", DEFAULT_MAX_PLANNER_REPAIR_ATTEMPTS
    )


def full_agent_reliability_settings() -> dict[str, int]:
    """Return bounded, environment-configurable full-agent safety settings."""
    return {
        "max_plan_goals": _positive_environment_int(
            "MAX_PLAN_GOALS", DEFAULT_MAX_PLAN_GOALS
        ),
        "max_goal_result_bytes": _positive_environment_int(
            "MAX_GOAL_RESULT_BYTES", DEFAULT_MAX_GOAL_RESULT_BYTES
        ),
        "max_goal_result_list_length": _positive_environment_int(
            "MAX_GOAL_RESULT_LIST_LENGTH", DEFAULT_MAX_GOAL_RESULT_LIST_LENGTH
        ),
        "max_goal_result_depth": _positive_environment_int(
            "MAX_GOAL_RESULT_DEPTH", DEFAULT_MAX_GOAL_RESULT_DEPTH
        ),
        "max_failure_family_attempts": _positive_environment_int(
            "MAX_FAILURE_FAMILY_ATTEMPTS", DEFAULT_MAX_FAILURE_FAMILY_ATTEMPTS
        ),
        "max_model_calls": _positive_environment_int(
            "MAX_MODEL_CALLS", DEFAULT_MAX_MODEL_CALLS
        ),
        "max_goal_rollbacks": _positive_environment_int(
            "MAX_GOAL_ROLLBACKS", DEFAULT_MAX_GOAL_ROLLBACKS
        ),
        "max_rollback_goals": _positive_environment_int(
            "MAX_ROLLBACK_GOALS", DEFAULT_MAX_ROLLBACK_GOALS
        ),
    }


def load_settings(*, load_dotenv_file: bool = True) -> Settings:
    """Load Nebius settings, raising a clear error for missing required values."""
    if load_dotenv_file:
        load_dotenv(override=False)

    api_key = os.getenv("NEBIUS_API_KEY")
    model = os.getenv("NEBIUS_MODEL")
    base_url = os.getenv("NEBIUS_BASE_URL") or DEFAULT_NEBIUS_BASE_URL
    max_code_repair_attempts, code_repair_no_progress_attempts = code_repair_settings()
    planner_repair_attempts = max_planner_repair_attempts()

    missing = [
        name
        for name, value in (("NEBIUS_API_KEY", api_key), ("NEBIUS_MODEL", model))
        if not value
    ]
    if missing:
        variables = ", ".join(missing)
        raise ConfigurationError(
            f"Missing required environment variable(s): {variables}"
        )

    return Settings(
        nebius_api_key=api_key,
        nebius_base_url=base_url,
        nebius_model=model,
        max_code_repair_attempts=max_code_repair_attempts,
        code_repair_no_progress_attempts=code_repair_no_progress_attempts,
        max_planner_repair_attempts=planner_repair_attempts,
    )
