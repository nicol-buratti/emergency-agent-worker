import json
from typing import Any
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_model: str = Field(default="gpt-4", validation_alias="LLM_MODEL")
    llm_api_key: SecretStr = Field(validation_alias="LLM_API_KEY")
    llm_base_url: str | None = Field(default=None, validation_alias="LLM_BASE_URL")
    extra_llm_models: list[str] | None = Field(
        default=None, validation_alias="EXTRA_LLM_MODELS"
    )

    @field_validator("extra_llm_models", mode="before")
    @classmethod
    def parse_extra_models(cls, value: Any) -> list[str] | None:
        if not value:
            return None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    raise ValueError("EXTRA_LLM_MODELS must be a JSON array.")
                return parsed
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse EXTRA_LLM_MODELS: {e}")
        return value
