"""Factory for an OpenAI-compatible Nebius Token Factory client."""

from openai import OpenAI

from data_analysis_agent.config import Settings


def create_nebius_client(settings: Settings) -> OpenAI:
    """Construct an OpenAI-compatible client without making an API request."""
    return OpenAI(
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
    )
