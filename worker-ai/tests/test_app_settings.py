import pytest
from pydantic import ValidationError

from src.app_settings import AppSettings


class TestAppSettingsDefaults:
    def test_requires_llm_api_key(self):
        """LLM_API_KEY has no default; omitting it must fail loudly at
        startup rather than the worker silently running without a key."""
        with pytest.raises(ValidationError) as exc_info:
            AppSettings()

        assert "llm_api_key" in str(exc_info.value).lower()

    def test_llm_model_defaults_to_gpt4(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")

        settings = AppSettings()

        assert settings.llm_model == "gpt-4"

    def test_llm_base_url_defaults_to_none(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")

        settings = AppSettings()

        assert settings.llm_base_url is None

    def test_extra_llm_models_defaults_to_none(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")

        settings = AppSettings()

        assert settings.extra_llm_models is None


class TestAppSettingsOverrides:
    def test_reads_values_from_environment(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "super-secret")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

        settings = AppSettings()

        assert settings.llm_model == "gpt-4o-mini"
        assert settings.llm_base_url == "https://openrouter.ai/api/v1"

    def test_api_key_is_kept_secret(self, monkeypatch):
        """SecretStr should mask the key in repr/str so it never ends up in
        logs, but the real value must still be retrievable when needed."""
        monkeypatch.setenv("LLM_API_KEY", "super-secret")

        settings = AppSettings()

        assert "super-secret" not in str(settings.llm_api_key)
        assert "super-secret" not in repr(settings.llm_api_key)
        assert settings.llm_api_key.get_secret_value() == "super-secret"

    def test_unknown_env_vars_are_ignored(self, monkeypatch):
        """model_config uses extra='ignore', so unrelated env vars (Redis,
        MQTT, etc.) must not break settings loading."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("REDIS_HOST", "some-host")
        monkeypatch.setenv("SOME_UNRELATED_VAR", "whatever")

        settings = AppSettings()

        assert not hasattr(settings, "redis_host")


class TestExtraLlmModelsParsing:
    def test_parses_valid_json_array(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("EXTRA_LLM_MODELS", '["gpt-4o-mini", "claude-3-haiku"]')

        settings = AppSettings()

        assert settings.extra_llm_models == ["gpt-4o-mini", "claude-3-haiku"]

    def test_empty_string_becomes_none(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("EXTRA_LLM_MODELS", "")

        settings = AppSettings()

        assert settings.extra_llm_models is None

    def test_already_a_list_passes_through(self, monkeypatch):
        """field_validator(mode='before') should also accept a value that
        is already a list (e.g. constructed programmatically in tests)."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")

        settings = AppSettings(extra_llm_models=["model-a", "model-b"])

        assert settings.extra_llm_models == ["model-a", "model-b"]

    def test_malformed_json_raises_clear_error(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("EXTRA_LLM_MODELS", "{not valid json")

        with pytest.raises(ValidationError) as exc_info:
            AppSettings()

        assert "EXTRA_LLM_MODELS" in str(exc_info.value)

    def test_json_object_instead_of_array_is_rejected(self, monkeypatch):
        """A JSON object parses fine as JSON but is not a list, and must
        still be rejected rather than silently accepted."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("EXTRA_LLM_MODELS", '{"model": "gpt-4o-mini"}')

        with pytest.raises(ValidationError):
            AppSettings()
