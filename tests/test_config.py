import pytest

from data_analysis_agent.config import (
    DEFAULT_MAX_CODE_REPAIR_ATTEMPTS,
    DEFAULT_NEBIUS_BASE_URL,
    ConfigurationError,
    load_settings,
)


def test_load_settings_uses_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_MODEL", "test-model")
    monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)

    settings = load_settings(load_dotenv_file=False)

    assert settings.nebius_base_url == DEFAULT_NEBIUS_BASE_URL


def test_load_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("NEBIUS_MODEL", "test-model")

    settings = load_settings(load_dotenv_file=False)

    assert settings.nebius_api_key == "test-key"
    assert settings.nebius_base_url == "https://example.test/v1/"
    assert settings.nebius_model == "test-model"


def test_load_settings_reads_mechanical_repair_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_MODEL", "test-model")
    monkeypatch.setenv("MAX_CODE_REPAIR_ATTEMPTS", "7")
    monkeypatch.setenv("CODE_REPAIR_NO_PROGRESS_ATTEMPTS", "2")

    settings = load_settings(load_dotenv_file=False)

    assert settings.max_code_repair_attempts == 7
    assert settings.code_repair_no_progress_attempts == 2


def test_load_settings_defaults_mechanical_repair_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_MODEL", "test-model")
    monkeypatch.delenv("MAX_CODE_REPAIR_ATTEMPTS", raising=False)

    assert load_settings(load_dotenv_file=False).max_code_repair_attempts == (
        DEFAULT_MAX_CODE_REPAIR_ATTEMPTS
    )


def test_load_settings_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setenv("NEBIUS_MODEL", "test-model")

    with pytest.raises(ConfigurationError, match="NEBIUS_API_KEY"):
        load_settings(load_dotenv_file=False)


def test_load_settings_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)

    with pytest.raises(ConfigurationError, match="NEBIUS_MODEL"):
        load_settings(load_dotenv_file=False)
