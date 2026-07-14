"""Environment-based configuration for Nebius Token Factory."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


class ConfigurationError(ValueError):
    """Raised when required Nebius configuration is missing."""


@dataclass(frozen=True)
class Settings:
    """Configuration required to construct a Nebius API client."""

    nebius_api_key: str
    nebius_base_url: str
    nebius_model: str


def load_settings(*, load_dotenv_file: bool = True) -> Settings:
    """Load Nebius settings, raising a clear error for missing required values."""
    if load_dotenv_file:
        load_dotenv(override=False)

    api_key = os.getenv("NEBIUS_API_KEY")
    model = os.getenv("NEBIUS_MODEL")
    base_url = os.getenv("NEBIUS_BASE_URL") or DEFAULT_NEBIUS_BASE_URL

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
    )
