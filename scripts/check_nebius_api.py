#!/usr/bin/env python3
"""Manually verify Nebius Token Factory configuration and connectivity."""

from __future__ import annotations

import sys

from openai import APIError, OpenAIError

from scientific_agent.config import ConfigurationError, load_settings
from scientific_agent.nebius_client import create_nebius_client


def main() -> int:
    """Run a minimal completion request and report whether it succeeds."""
    try:
        settings = load_settings()
        client = create_nebius_client(settings)
        response = client.chat.completions.create(
            model=settings.nebius_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Reply with exactly: API connection successful",
                },
            ],
        )
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1
    except (APIError, OpenAIError) as error:
        print(f"Nebius API check failed: {error}", file=sys.stderr)
        return 1

    print(f"Model: {settings.nebius_model}")
    print(f"Base URL: {settings.nebius_base_url}")
    print(response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
