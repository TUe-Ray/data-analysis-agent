import importlib
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from data_analysis_agent import nebius_client
from data_analysis_agent.config import Settings
from data_analysis_agent.models import (
    EmptyModelResponseError,
    MalformedModelResponseError,
    ModelCapabilityError,
    ModelOutputLimitError,
    NebiusRoleModel,
)
from data_analysis_agent.nebius_client import create_nebius_client
from data_analysis_agent.schemas import PythonGeneration


def _response(
    content: str | None,
    *,
    finish_reason: str | None = "stop",
    usage: tuple[int, int, int] | None = None,
    tool_calls: list[object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="response-id",
        model="test-model",
        usage=(
            SimpleNamespace(
                prompt_tokens=usage[0],
                completion_tokens=usage[1],
                total_tokens=usage[2],
            )
            if usage is not None
            else None
        ),
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    refusal=None,
                    tool_calls=tool_calls,
                ),
            )
        ],
    )


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


def test_nebius_uses_purpose_specific_output_limits() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _response('{"ok":true}')
    model = NebiusRoleModel(client=client, model="test-model")

    model.generate(role="planner", messages=[])
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 8192
    model.generate(role="executor", messages=[])
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 8192
    model.generate(role="verifier", messages=[])
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 8192
    model.generate(role="direct_answer", messages=[])
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 4096
    model.generate_structured(
        role="executor",
        messages=[],
        schema_name="python_generation",
        schema=PythonGeneration.model_json_schema(),
    )
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 32768


def test_legacy_general_limit_does_not_lower_python_limit() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _response('{"ok":true}')
    model = NebiusRoleModel(client=client, model="test-model", max_output_tokens=5000)

    model.generate(role="planner", messages=[])
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 5000
    model.generate_structured(
        role="executor",
        messages=[],
        schema_name="python_repair",
        schema=PythonGeneration.model_json_schema(),
    )
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 32768


def test_nebius_retries_empty_responses_and_accumulates_usage() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        _response(None, usage=(10, 1, 11)),
        _response("   ", usage=(20, 2, 22)),
        _response('{"ok":true}', usage=(30, 3, 33)),
    ]
    model = NebiusRoleModel(client=client, model="test-model")

    with patch("data_analysis_agent.models.time.sleep") as sleep:
        output = model.generate(role="executor", messages=[])

    assert output == '{"ok":true}'
    assert client.chat.completions.create.call_count == 3
    assert model.last_api_request_count == 3
    assert model.last_response_retry_count == 2
    assert model.last_token_usage == {
        "prompt_tokens": 60,
        "completion_tokens": 6,
        "total_tokens": 66,
    }
    assert len(model.last_provider_attempts) == 3
    assert model.last_provider_attempts[0]["content_length"] == 0
    sleep.assert_has_calls([call(0.5), call(1.5)])


def test_nebius_raises_typed_error_after_empty_response_retries() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _response(None)
    model = NebiusRoleModel(client=client, model="test-model")

    with (
        patch("data_analysis_agent.models.time.sleep"),
        pytest.raises(EmptyModelResponseError, match="after 3 attempts"),
    ):
        model.generate(role="executor", messages=[])

    assert client.chat.completions.create.call_count == 3
    assert model.last_response_retry_count == 2


def test_nebius_retries_tool_call_only_response_with_correction() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        _response(None, finish_reason="tool_calls", tool_calls=[object()]),
        _response('{"strategy":"generated_python"}'),
    ]
    model = NebiusRoleModel(client=client, model="test-model")

    with patch("data_analysis_agent.models.time.sleep") as sleep:
        output = model.generate(
            role="executor", messages=[{"role": "user", "content": "x"}]
        )

    assert output == '{"strategy":"generated_python"}'
    assert model.last_response_retry_count == 1
    assert client.chat.completions.create.call_count == 2
    retry_messages = client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert retry_messages[-1]["role"] == "user"
    assert "No tools are available" in retry_messages[-1]["content"]
    assert model.last_provider_attempts[0]["tool_calls"] is True
    sleep.assert_called_once_with(0.5)


def test_nebius_stops_after_repeated_tool_call_only_responses() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _response(
        None, finish_reason="tool_calls", tool_calls=[object()]
    )
    model = NebiusRoleModel(client=client, model="test-model")

    with (
        patch("data_analysis_agent.models.time.sleep"),
        pytest.raises(MalformedModelResponseError, match="after 3 attempts"),
    ):
        model.generate(role="executor", messages=[])

    assert client.chat.completions.create.call_count == 3
    assert model.last_response_retry_count == 2


def test_nebius_does_not_retry_blank_length_terminated_response() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _response(
        None, finish_reason="length"
    )
    model = NebiusRoleModel(client=client, model="test-model")

    with pytest.raises(ModelOutputLimitError, match="finish_reason='length'"):
        model.generate(role="planner", messages=[])

    assert client.chat.completions.create.call_count == 1
    assert model.last_response_retry_count == 0


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


def test_generic_structured_400_does_not_poison_capability_cache() -> None:
    NebiusRoleModel.clear_capability_cache()
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    context_error = BadRequestError(
        "maximum context length exceeded",
        response=httpx.Response(400, request=request),
        body={"error": {"message": "maximum context length exceeded"}},
    )
    client = Mock()
    client.base_url = "https://provider-a.test/v1"
    client.chat.completions.create.side_effect = [
        context_error,
        _response('{"ok":true}'),
    ]
    model = NebiusRoleModel(client=client, model="shared-model")
    arguments = {
        "role": "executor",
        "messages": [],
        "schema_name": "python_generation",
        "schema": PythonGeneration.model_json_schema(),
    }

    with pytest.raises(BadRequestError, match="context length"):
        model.generate_structured(**arguments)
    assert model.generate_structured(**arguments) == '{"ok":true}'
    assert client.chat.completions.create.call_count == 2


def test_schema_capability_cache_is_scoped_to_base_url_and_model() -> None:
    NebiusRoleModel.clear_capability_cache()
    request = httpx.Request("POST", "https://provider-a.test/v1/chat/completions")
    unsupported = BadRequestError(
        "json_schema structured output is unsupported",
        response=httpx.Response(400, request=request),
        body={"error": "json_schema unsupported"},
    )
    first = Mock()
    first.base_url = "https://provider-a.test/v1/"
    first.chat.completions.create.side_effect = unsupported
    second = Mock()
    second.base_url = "https://provider-b.test/v1/"
    second.chat.completions.create.return_value = _response('{"ok":true}')
    arguments = {
        "role": "executor",
        "messages": [],
        "schema_name": "python_generation",
        "schema": PythonGeneration.model_json_schema(),
    }

    with pytest.raises(ModelCapabilityError):
        NebiusRoleModel(client=first, model="shared-model").generate_structured(
            **arguments
        )
    assert (
        NebiusRoleModel(client=second, model="shared-model").generate_structured(
            **arguments
        )
        == '{"ok":true}'
    )
    assert second.chat.completions.create.call_count == 1
