import importlib
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from data_analysis_agent import nebius_client
from data_analysis_agent.config import Settings
from data_analysis_agent.models import ModelCapabilityError, NebiusRoleModel
from data_analysis_agent.nebius_client import create_nebius_client
from data_analysis_agent.schemas import PythonGeneration


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


def test_nebius_structured_generation_uses_strict_json_schema() -> None:
    NebiusRoleModel.clear_capability_cache()
    client = Mock()
    response = Mock()
    response.usage = None
    response.choices = [
        Mock(message=Mock(content='{"kind":"python","code":"x=1","summary":""}'))
    ]
    client.chat.completions.create.return_value = response
    schema = PythonGeneration.model_json_schema()
    model = NebiusRoleModel(client=client, model="structured-model")

    model.generate_structured(
        role="executor",
        messages=[],
        schema_name="python_generation",
        schema=schema,
    )

    assert client.chat.completions.create.call_args.kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "python_generation",
            "strict": True,
            "schema": schema,
        },
    }


def test_unsupported_schema_is_cached_per_configured_model() -> None:
    NebiusRoleModel.clear_capability_cache()
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    rejection = BadRequestError(
        "response_format is unsupported",
        response=httpx.Response(400, request=request),
        body={"error": "unsupported"},
    )
    rejected_client = Mock()
    rejected_client.chat.completions.create.side_effect = rejection
    rejected = NebiusRoleModel(client=rejected_client, model="unsupported-model")
    arguments = {
        "role": "executor",
        "messages": [],
        "schema_name": "python_generation",
        "schema": PythonGeneration.model_json_schema(),
    }

    with pytest.raises(ModelCapabilityError, match="model_capability_error"):
        rejected.generate_structured(**arguments)
    with pytest.raises(ModelCapabilityError, match="does not support"):
        rejected.generate_structured(**arguments)
    assert rejected_client.chat.completions.create.call_count == 1

    supported_client = Mock()
    response = Mock()
    response.usage = None
    response.choices = [
        Mock(message=Mock(content='{"kind":"python","code":"x=1","summary":""}'))
    ]
    supported_client.chat.completions.create.return_value = response
    changed_model = NebiusRoleModel(client=supported_client, model="supported-model")

    changed_model.generate_structured(**arguments)

    assert supported_client.chat.completions.create.call_count == 1
