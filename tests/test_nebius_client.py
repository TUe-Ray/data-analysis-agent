import importlib
from unittest.mock import Mock, patch

from data_analysis_agent import nebius_client
from data_analysis_agent.config import Settings
from data_analysis_agent.nebius_client import create_nebius_client


def test_importing_client_module_does_not_construct_a_client() -> None:
    with patch("openai.OpenAI") as client_constructor:
        importlib.reload(nebius_client)
        client_constructor.assert_not_called()

    importlib.reload(nebius_client)


def test_create_nebius_client_uses_settings() -> None:
    settings = Settings(
        nebius_api_key="test-key",
        nebius_base_url="https://example.test/v1/",
        nebius_model="test-model",
    )
    client_constructor = Mock()

    with patch("data_analysis_agent.nebius_client.OpenAI", client_constructor):
        client = create_nebius_client(settings)

    assert client is client_constructor.return_value
    client_constructor.assert_called_once_with(
        api_key="test-key",
        base_url="https://example.test/v1/",
    )
