import importlib
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import APIConnectionError

from data_analysis_agent import nebius_client
from data_analysis_agent.config import Settings
from data_analysis_agent.models import NebiusRoleModel
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
        max_retries=0,
    )


def test_nebius_model_retries_one_transient_connection_failure() -> None:
    client = Mock()
    response = Mock()
    response.usage = None
    response.choices = [Mock(message=Mock(content='{"ok":true}'))]
    connection_error = APIConnectionError(
        request=httpx.Request("POST", "https://example.test/v1/chat/completions")
    )
    client.chat.completions.create.side_effect = [connection_error, response]
    model = NebiusRoleModel(client=client, model="test-model", temperature=0)

    output = model.generate(role="direct_answer", messages=[])

    assert output == '{"ok":true}'
    assert client.chat.completions.create.call_count == 2
    assert model.last_api_request_count == 2
    assert model.last_transport_retry_count == 1


def test_nebius_model_stops_after_one_transport_retry() -> None:
    client = Mock()
    errors = [
        APIConnectionError(
            request=httpx.Request("POST", "https://example.test/v1/chat/completions")
        )
        for _ in range(2)
    ]
    client.chat.completions.create.side_effect = errors
    model = NebiusRoleModel(client=client, model="test-model", temperature=0)

    with pytest.raises(APIConnectionError):
        model.generate(role="direct_answer", messages=[])

    assert client.chat.completions.create.call_count == 2
    assert model.last_api_request_count == 2
    assert model.last_transport_retry_count == 1
