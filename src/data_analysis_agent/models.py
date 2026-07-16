"""Small role-model adapters for scripted and live Nebius execution."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Protocol

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

Role = Literal[
    "planner",
    "executor",
    "verifier",
    "direct_answer",
    "one_shot_code",
    "single_agent",
    "final_checker",
]


class RoleModel(Protocol):
    """Minimal interface shared by offline and live role models."""

    def generate(self, *, role: Role, messages: list[dict[str, str]]) -> str:
        """Generate one response for a role-specific message context."""
        ...

    def generate_structured(
        self,
        *,
        role: Role,
        messages: list[dict[str, str]],
        schema_name: str,
        schema: dict[str, object],
    ) -> str:
        """Generate a response under a provider-enforced strict JSON Schema."""
        ...


class ModelCapabilityError(RuntimeError):
    """Raised when a configured model cannot enforce required structured output."""


class ProviderResponseError(RuntimeError):
    """Base class for usable-content failures after a successful API response."""


class EmptyModelResponseError(ProviderResponseError):
    """Raised when bounded retries cannot obtain final model content."""


class ModelOutputLimitError(ProviderResponseError):
    """Raised when a blank response ended because its output limit was reached."""


class ModelRefusalError(ProviderResponseError):
    """Raised when a provider reports a refusal instead of final content."""


class MalformedModelResponseError(ProviderResponseError):
    """Raised when a provider never returns usable final response content."""


class ModelCallLimitError(RuntimeError):
    """Raised before a full-agent run exceeds its logical model-call ceiling."""


DEFAULT_ORDINARY_OUTPUT_TOKENS = 8192
DEFAULT_PYTHON_OUTPUT_TOKENS = 32768
DEFAULT_LEGACY_OUTPUT_TOKENS = 4096
_PYTHON_SCHEMA_NAMES = frozenset(
    {
        "python_generation",
        "python_repair",
        "single_agent_python_generation",
        "single_agent_python_repair",
    }
)


@dataclass(frozen=True)
class ModelCall:
    """One recorded scripted-model call, exposed for deterministic tests."""

    role: Role
    messages: list[dict[str, str]]
    structured_schema_name: str | None = None


@dataclass(frozen=True)
class ModelExchange:
    """One complete model exchange retained by the demo logging layer."""

    role: Role
    messages: list[dict[str, str]]
    response: str | None
    latency_seconds: float
    error: str | None = None
    token_usage: dict[str, int] | None = None
    api_request_count: int = 1
    transport_retry_count: int = 0
    response_retry_count: int = 0
    finish_reason: str | None = None
    purpose: str | None = None
    structured_schema_name: str | None = None
    provider_attempts: list[dict[str, object]] = field(default_factory=list)


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
            "single_agent": 0,
            "final_checker": 0,
        }
        self.calls: list[ModelCall] = []

    def generate(self, *, role: Role, messages: list[dict[str, str]]) -> str:
        """Return the next scripted response and record the isolated context."""
        return self._next(role=role, messages=messages, schema_name=None)

    def generate_structured(
        self,
        *,
        role: Role,
        messages: list[dict[str, str]],
        schema_name: str,
        schema: dict[str, object],
    ) -> str:
        """Use scripted JSON while recording that a strict contract was requested."""
        del schema
        return self._next(role=role, messages=messages, schema_name=schema_name)

    def _next(
        self,
        *,
        role: Role,
        messages: list[dict[str, str]],
        schema_name: str | None,
    ) -> str:
        self.calls.append(
            ModelCall(
                role=role,
                messages=[dict(message) for message in messages],
                structured_schema_name=schema_name,
            )
        )
        position = self._positions[role]
        responses = self._responses.get(role, [])
        if position >= len(responses):
            raise RuntimeError(f"No scripted {role} response remains")
        self._positions[role] = position + 1
        response = responses[position]
        # Historical deterministic fixtures used a single ``code`` field.  Keep
        # that adapter confined to the scripted test double; the live client is
        # given only the line-oriented production JSON Schema and never accepts
        # this raw-string fallback.
        if schema_name in {
            "python_generation",
            "python_repair",
            "single_agent_python_generation",
            "single_agent_python_repair",
        }:
            response = _adapt_legacy_scripted_python_contract(response)
        return response


def _adapt_legacy_scripted_python_contract(response: str) -> str:
    """Convert old in-repository scripted fixtures, without relaxing live IO."""
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return response
    if not isinstance(parsed, dict) or "code" not in parsed or "code_lines" in parsed:
        return response
    code = parsed.pop("code")
    if not isinstance(code, str):
        return response
    parsed["code_lines"] = code.splitlines()
    return json.dumps(parsed, ensure_ascii=False)


class NebiusRoleModel:
    """Use one configured Nebius model for each V0 role."""

    _json_schema_capabilities: ClassVar[dict[tuple[str, str], bool]] = {}

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        planner_max_output_tokens: int | None = None,
        executor_max_output_tokens: int | None = None,
        verifier_max_output_tokens: int | None = None,
        final_checker_max_output_tokens: int | None = None,
        python_max_output_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._max_output_tokens = max_output_tokens
        self._planner_max_output_tokens = planner_max_output_tokens
        self._executor_max_output_tokens = executor_max_output_tokens
        self._verifier_max_output_tokens = verifier_max_output_tokens
        self._final_checker_max_output_tokens = final_checker_max_output_tokens
        self._python_max_output_tokens = python_max_output_tokens
        self.last_token_usage: dict[str, int] | None = None
        self.last_api_request_count = 0
        self.last_transport_retry_count = 0
        self.last_response_retry_count = 0
        self.last_finish_reason: str | None = None
        self.last_provider_attempts: list[dict[str, object]] = []

    def generate(self, *, role: Role, messages: list[dict[str, str]]) -> str:
        """Send a role-specific request through the OpenAI-compatible client."""
        return self._request(
            messages=messages,
            role=role,
            purpose=self._purpose(role=role, schema_name=None),
        )

    def generate_structured(
        self,
        *,
        role: Role,
        messages: list[dict[str, str]],
        schema_name: str,
        schema: dict[str, object],
    ) -> str:
        """Require strict provider-native JSON Schema for executable responses."""
        capability_key = self._capability_key()
        if self._json_schema_capabilities.get(capability_key) is False:
            raise ModelCapabilityError(
                "model_capability_error: model "
                f"{self._model!r} does not support strict json_schema output"
            )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }
        try:
            content = self._request(
                messages=messages,
                response_format=response_format,
                role=role,
                purpose=self._purpose(role=role, schema_name=schema_name),
            )
        except APIStatusError as error:
            if self._is_explicit_json_schema_unsupported(error):
                self._json_schema_capabilities[capability_key] = False
                raise ModelCapabilityError(
                    "model_capability_error: model "
                    f"{self._model!r} rejected strict json_schema output"
                ) from error
            raise
        self._json_schema_capabilities[capability_key] = True
        return content

    def _capability_key(self) -> tuple[str, str]:
        """Scope structured-output facts to the provider endpoint and model."""
        raw_url = getattr(self._client, "base_url", None)
        if isinstance(raw_url, str):
            base_url = raw_url
        elif raw_url is None or type(raw_url).__module__.startswith("unittest.mock"):
            base_url = ""
        else:
            base_url = str(raw_url)
        return (base_url.rstrip("/").lower(), self._model)

    @staticmethod
    def _is_explicit_json_schema_unsupported(error: APIStatusError) -> bool:
        """Only capability-specific provider rejections may poison the cache."""
        if error.status_code not in {400, 422}:
            return False
        details = [str(error), str(getattr(error, "body", ""))]
        response = getattr(error, "response", None)
        if response is not None:
            try:
                details.append(response.text)
            except Exception:  # pragma: no cover - defensive SDK compatibility
                pass
        text = " ".join(details).lower()
        capability_marker = any(
            marker in text
            for marker in (
                "response_format",
                "json_schema",
                "json schema",
                "strict structured",
                "structured output",
                "structured outputs",
            )
        )
        unsupported_marker = any(
            marker in text
            for marker in (
                "unsupported",
                "not supported",
                "does not support",
                "not available",
                "unrecognized parameter",
                "unknown parameter",
            )
        )
        return capability_marker and unsupported_marker

    @classmethod
    def clear_capability_cache(cls) -> None:
        """Clear cached provider capability facts (primarily for isolated tests)."""
        cls._json_schema_capabilities.clear()

    def _request(
        self,
        *,
        messages: list[dict[str, str]],
        role: Role,
        purpose: str,
        response_format: dict[str, object] | None = None,
    ) -> str:
        self.last_token_usage = None
        self.last_api_request_count = 0
        self.last_transport_retry_count = 0
        self.last_response_retry_count = 0
        self.last_finish_reason = None
        self.last_provider_attempts = []
        request_messages = list(messages)
        arguments: dict[str, object] = {
            "model": self._model,
            "messages": request_messages,
        }
        if self._temperature is not None:
            arguments["temperature"] = self._temperature
        if self._top_p is not None:
            arguments["top_p"] = self._top_p
        max_output_tokens = self._output_token_limit(role=role, purpose=purpose)
        arguments["max_tokens"] = max_output_tokens
        if response_format is not None:
            arguments["response_format"] = response_format
        last_diagnostic: dict[str, object] = {}
        tool_call_correction_added = False
        for response_attempt in range(3):
            response = self._request_with_transport(
                arguments=arguments,
                response_attempt=response_attempt,
                purpose=purpose,
                max_output_tokens=max_output_tokens,
            )
            diagnostic, content, exception_type = self._inspect_response(
                response=response,
                response_attempt=response_attempt,
                purpose=purpose,
                max_output_tokens=max_output_tokens,
            )
            self.last_provider_attempts.append(diagnostic)
            self._accumulate_usage(response)
            finish_reason = diagnostic.get("finish_reason")
            self.last_finish_reason = (
                str(finish_reason) if isinstance(finish_reason, str) else None
            )
            if content is not None:
                return content
            last_diagnostic = diagnostic
            retryable = exception_type is EmptyModelResponseError or (
                exception_type is MalformedModelResponseError
                and diagnostic["tool_calls"] is True
            )
            if not retryable:
                raise exception_type(self._response_error_message(diagnostic))
            if response_attempt == 2:
                raise exception_type(
                    self._response_error_message(diagnostic, attempts=3)
                )
            if diagnostic["tool_calls"] is True and not tool_call_correction_added:
                request_messages = [
                    *request_messages,
                    {
                        "role": "user",
                        "content": (
                            "Your previous response attempted an unsupported tool "
                            "call and provided no final answer. No tools are "
                            "available. Return the required final response directly "
                            "and do not make tool calls."
                        ),
                    },
                ]
                arguments["messages"] = request_messages
                tool_call_correction_added = True
            self.last_response_retry_count += 1
            time.sleep((0.5, 1.5)[response_attempt])
        raise EmptyModelResponseError(self._response_error_message(last_diagnostic))

    def _request_with_transport(
        self,
        *,
        arguments: dict[str, object],
        response_attempt: int,
        purpose: str,
        max_output_tokens: int,
    ) -> object:
        for transport_attempt in range(2):
            self.last_api_request_count += 1
            try:
                return self._client.chat.completions.create(**arguments)
            except (APIConnectionError, APITimeoutError) as error:
                self.last_provider_attempts.append(
                    self._transport_failure_diagnostic(
                        error=error,
                        response_attempt=response_attempt,
                        transport_attempt=transport_attempt,
                        purpose=purpose,
                        max_output_tokens=max_output_tokens,
                    )
                )
                if transport_attempt == 1:
                    raise
                self.last_transport_retry_count += 1
            except APIStatusError as error:
                self.last_provider_attempts.append(
                    self._transport_failure_diagnostic(
                        error=error,
                        response_attempt=response_attempt,
                        transport_attempt=transport_attempt,
                        purpose=purpose,
                        max_output_tokens=max_output_tokens,
                    )
                )
                if transport_attempt == 1 or error.status_code < 500:
                    raise
                self.last_transport_retry_count += 1
        raise AssertionError("transport retry loop exited without a response")

    def _purpose(self, *, role: Role, schema_name: str | None) -> str:
        if schema_name is not None:
            return schema_name
        return "executor_strategy" if role == "executor" else role

    def _output_token_limit(self, *, role: Role, purpose: str) -> int:
        if purpose in _PYTHON_SCHEMA_NAMES:
            return self._python_max_output_tokens or DEFAULT_PYTHON_OUTPUT_TOKENS
        if role == "planner":
            return (
                self._planner_max_output_tokens
                or self._max_output_tokens
                or DEFAULT_ORDINARY_OUTPUT_TOKENS
            )
        if role == "executor":
            return (
                self._executor_max_output_tokens
                or self._max_output_tokens
                or DEFAULT_ORDINARY_OUTPUT_TOKENS
            )
        if role == "verifier":
            return (
                self._verifier_max_output_tokens
                or self._max_output_tokens
                or DEFAULT_ORDINARY_OUTPUT_TOKENS
            )
        if role == "final_checker":
            return (
                self._final_checker_max_output_tokens
                or self._verifier_max_output_tokens
                or self._max_output_tokens
                or DEFAULT_ORDINARY_OUTPUT_TOKENS
            )
        return self._max_output_tokens or DEFAULT_LEGACY_OUTPUT_TOKENS

    def _inspect_response(
        self,
        *,
        response: object,
        response_attempt: int,
        purpose: str,
        max_output_tokens: int,
    ) -> tuple[dict[str, object], str | None, type[ProviderResponseError]]:
        choices = getattr(response, "choices", None)
        choice_count = len(choices) if isinstance(choices, (list, tuple)) else 0
        choice = choices[0] if choice_count else None
        message = getattr(choice, "message", None) if choice is not None else None
        content = getattr(message, "content", None) if message is not None else None
        content_text = content if isinstance(content, str) else None
        stripped = content_text.strip() if content_text is not None else ""
        finish_reason = getattr(choice, "finish_reason", None) if choice else None
        finish_reason_text = (
            str(finish_reason) if isinstance(finish_reason, str) else None
        )
        refusal = getattr(message, "refusal", None) if message is not None else None
        tool_calls = (
            getattr(message, "tool_calls", None) if message is not None else None
        )
        has_refusal = isinstance(refusal, str) and bool(refusal.strip())
        has_tool_calls = isinstance(tool_calls, (list, tuple)) and bool(tool_calls)
        diagnostic: dict[str, object] = {
            "request_attempt": self.last_api_request_count,
            "response_retry_number": response_attempt,
            "transport_retry_count": self.last_transport_retry_count,
            "response_id": _safe_text_attribute(response, "id"),
            "returned_model": _safe_text_attribute(response, "model"),
            "choice_count": choice_count,
            "selected_choice_index": 0 if choice is not None else None,
            "finish_reason": finish_reason_text,
            "message_present": message is not None,
            "content_is_none": content is None,
            "content_length": len(stripped),
            "refusal": has_refusal,
            "tool_calls": has_tool_calls,
            "reasoning_fields_present": any(
                getattr(item, field_name, None) is not None
                for item in (choice, message)
                if item is not None
                for field_name in ("reasoning", "reasoning_content")
            ),
            "prompt_tokens": _usage_value(response, "prompt_tokens"),
            "completion_tokens": _usage_value(response, "completion_tokens"),
            "total_tokens": _usage_value(response, "total_tokens"),
            "max_output_tokens": max_output_tokens,
            "purpose": purpose,
        }
        if stripped:
            return diagnostic, content_text, EmptyModelResponseError
        normalized_reason = (finish_reason_text or "").lower()
        if normalized_reason in {"length", "max_tokens", "max_output_tokens"}:
            return diagnostic, None, ModelOutputLimitError
        if has_refusal:
            return diagnostic, None, ModelRefusalError
        if has_tool_calls:
            return diagnostic, None, MalformedModelResponseError
        return diagnostic, None, EmptyModelResponseError

    def _accumulate_usage(self, response: object) -> None:
        usage = {
            key: _usage_value(response, key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        if not any(value is not None for value in usage.values()):
            return
        existing = self.last_token_usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.last_token_usage = {
            key: existing[key] + (value or 0) for key, value in usage.items()
        }

    def _transport_failure_diagnostic(
        self,
        *,
        error: Exception,
        response_attempt: int,
        transport_attempt: int,
        purpose: str,
        max_output_tokens: int,
    ) -> dict[str, object]:
        return {
            "request_attempt": self.last_api_request_count,
            "response_retry_number": response_attempt,
            "transport_retry_count": self.last_transport_retry_count,
            "transport_attempt": transport_attempt,
            "purpose": purpose,
            "max_output_tokens": max_output_tokens,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        }

    def _response_error_message(
        self, diagnostic: dict[str, object], *, attempts: int | None = None
    ) -> str:
        attempt_text = f" after {attempts} attempts" if attempts is not None else ""
        return (
            "Nebius returned no usable final content"
            f"{attempt_text} (finish_reason={diagnostic.get('finish_reason')!r}, "
            f"choices={diagnostic.get('choice_count')}, "
            f"content_length={diagnostic.get('content_length')}, "
            f"tool_calls={diagnostic.get('tool_calls')}, "
            f"refusal={diagnostic.get('refusal')}, "
            f"max_output_tokens={diagnostic.get('max_output_tokens')})"
        )


def _safe_text_attribute(value: object, name: str) -> str | None:
    attribute = getattr(value, name, None)
    return attribute if isinstance(attribute, str) else None


def _usage_value(response: object, name: str) -> int | None:
    usage = getattr(response, "usage", None)
    value = getattr(usage, name, None) if usage is not None else None
    return value if isinstance(value, int) else None


class RecordingRoleModel:
    """Record exact role messages and raw responses for local run logs."""

    def __init__(
        self,
        model: RoleModel,
        call_observer: Callable[[str, Role, int, float, str | None], None]
        | None = None,
        max_calls: int | None = None,
    ) -> None:
        self._model = model
        self._call_observer = call_observer
        self._max_calls = max_calls
        self.exchanges: list[ModelExchange] = []

    def generate(self, *, role: Role, messages: list[dict[str, str]]) -> str:
        """Delegate generation and retain a secret-free exchange record."""
        return self._record(
            role=role,
            messages=messages,
            schema_name=None,
            generate=lambda: self._model.generate(role=role, messages=messages),
        )

    def generate_structured(
        self,
        *,
        role: Role,
        messages: list[dict[str, str]],
        schema_name: str,
        schema: dict[str, object],
    ) -> str:
        """Record a strict structured generation without weakening its contract."""
        return self._record(
            role=role,
            messages=messages,
            schema_name=schema_name,
            generate=lambda: self._model.generate_structured(
                role=role,
                messages=messages,
                schema_name=schema_name,
                schema=schema,
            ),
        )

    def _record(
        self,
        *,
        role: Role,
        messages: list[dict[str, str]],
        schema_name: str | None,
        generate: Callable[[], str],
    ) -> str:
        if self._max_calls is not None and len(self.exchanges) >= self._max_calls:
            raise ModelCallLimitError(
                f"full-agent model-call ceiling reached ({self._max_calls})"
            )
        started = time.perf_counter()
        call_number = len(self.exchanges) + 1
        if self._call_observer:
            self._call_observer("start", role, call_number, 0.0, None)
        copied_messages = [dict(message) for message in messages]
        try:
            response = generate()
        except Exception as error:
            elapsed = time.perf_counter() - started
            error_text = f"{type(error).__name__}: {error}"
            self.exchanges.append(
                ModelExchange(
                    role=role,
                    messages=copied_messages,
                    response=None,
                    latency_seconds=elapsed,
                    error=error_text,
                    token_usage=_token_usage(self._model),
                    api_request_count=_request_count(self._model),
                    transport_retry_count=_retry_count(self._model),
                    response_retry_count=_response_retry_count(self._model),
                    finish_reason=_finish_reason(self._model),
                    purpose=_purpose(role=role, schema_name=schema_name),
                    structured_schema_name=schema_name,
                    provider_attempts=_provider_attempts(self._model),
                )
            )
            if self._call_observer:
                self._call_observer("end", role, call_number, elapsed, error_text)
            raise
        elapsed = time.perf_counter() - started
        self.exchanges.append(
            ModelExchange(
                role=role,
                messages=copied_messages,
                response=response,
                latency_seconds=elapsed,
                token_usage=_token_usage(self._model),
                api_request_count=_request_count(self._model),
                transport_retry_count=_retry_count(self._model),
                response_retry_count=_response_retry_count(self._model),
                finish_reason=_finish_reason(self._model),
                purpose=_purpose(role=role, schema_name=schema_name),
                structured_schema_name=schema_name,
                provider_attempts=_provider_attempts(self._model),
            )
        )
        if self._call_observer:
            self._call_observer("end", role, call_number, elapsed, None)
        return response


def _token_usage(model: RoleModel) -> dict[str, int] | None:
    """Snapshot optional provider usage without requiring it from fake models."""
    usage = getattr(model, "last_token_usage", None)
    return dict(usage) if isinstance(usage, dict) else None


def _request_count(model: RoleModel) -> int:
    value = getattr(model, "last_api_request_count", 1)
    return value if isinstance(value, int) and value > 0 else 1


def _retry_count(model: RoleModel) -> int:
    value = getattr(model, "last_transport_retry_count", 0)
    return value if isinstance(value, int) and value >= 0 else 0


def _response_retry_count(model: RoleModel) -> int:
    value = getattr(model, "last_response_retry_count", 0)
    return value if isinstance(value, int) and value >= 0 else 0


def _finish_reason(model: RoleModel) -> str | None:
    value = getattr(model, "last_finish_reason", None)
    return value if isinstance(value, str) else None


def _provider_attempts(model: RoleModel) -> list[dict[str, object]]:
    attempts = getattr(model, "last_provider_attempts", [])
    return [dict(item) for item in attempts if isinstance(item, dict)]


def _purpose(*, role: Role, schema_name: str | None) -> str:
    if schema_name is not None:
        return schema_name
    return "executor_strategy" if role == "executor" else role


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
with open({str(data_path)!r}, encoding="utf-8") as handle:
    values = [float(row["value"]) for row in csv.DictReader(handle) if row["value"]]
differences = [abs(right - left) for left, right in zip(values, values[1:])]
result = sum(differences) / len(differences)
__agent_result__ = {{"mean_absolute_successive_difference": result}}
"""
    wrong_column_code = f"""import csv
with open({str(data_path)!r}, encoding="utf-8") as handle:
    values = [float(row["wrong_value"]) for row in csv.DictReader(handle)]
__agent_result__ = {{"mean_absolute_successive_difference": values[0]}}
"""
    other_wrong_code = wrong_column_code.replace("wrong_value", "still_wrong")

    def generation(code: str) -> str:
        return json.dumps(
            {"kind": "python", "code": code, "summary": "Compute the goal."}
        )

    def repair(code: str) -> str:
        return json.dumps(
            {
                "kind": "python_repair",
                "code": code,
                "summary": "Repair the runtime failure.",
                "addressed_failure_category": "runtime_error",
            }
        )

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
                "executor": [python_strategy, generation(good_code)],
                "verifier": [pass_python],
            }
        )
    if scenario == "generated-python-repair":
        return ScriptedRoleModel(
            {
                "planner": [structured_python_plan],
                "executor": [
                    python_strategy,
                    generation(wrong_column_code),
                    repair(good_code),
                ],
                "verifier": [pass_python],
            }
        )
    if scenario == "generated-python-failure":
        return ScriptedRoleModel(
            {
                "planner": [structured_python_plan, structured_python_plan],
                "executor": [
                    python_strategy,
                    generation(wrong_column_code),
                    repair(other_wrong_code),
                    repair(wrong_column_code),
                    repair(wrong_column_code),
                    repair(wrong_column_code),
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
