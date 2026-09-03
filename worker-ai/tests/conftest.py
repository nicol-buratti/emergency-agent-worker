import sys
from pathlib import Path

import pytest

# Make the worker-ai package importable when running `pytest` from the
# worker-ai/ directory or from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Ensure tests never read the developer's real .env file and always
    start from a clean, known environment. Individual tests can still
    override specific variables with monkeypatch.setenv(...).
    """
    # Point pydantic-settings at an empty .env so a real one on disk
    # (with real secrets) never leaks into a test run.
    empty_env = tmp_path / ".env"
    empty_env.write_text("")
    monkeypatch.chdir(tmp_path)

    for var in [
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "EXTRA_LLM_MODELS",
        "REDIS_HOST",
        "REDIS_PORT",
        "QUEUE_NAME",
        "MQTT_BROKER_ADDRESS",
        "MQTT_PUBLISH_TOPIC",
        "LANGFUSE_ENABLED",
    ]:
        monkeypatch.delenv(var, raising=False)

    yield


@pytest.fixture
def valid_assessment_payload():
    """A minimal, schema-valid ThreatAssessment payload usable across tests."""
    return {
        "room": "room-101",
        "warning": "none",
        "danger": "none",
        "danger_type": "none",
        "danger_score": 0.0,
        "justification": "No anomalies detected in telemetry or graph data.",
    }
