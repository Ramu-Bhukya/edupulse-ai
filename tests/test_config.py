import pytest

from edupulse import config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_TEMPERATURE", raising=False)


def test_missing_key_raises():
    with pytest.raises(config.MissingAPIKeyError):
        config.get_api_key()


def test_env_key_is_used(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "  sk-from-env  ")
    assert config.get_api_key() == "sk-from-env"


def test_override_wins_over_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert config.get_api_key("sk-typed-in") == "sk-typed-in"


def test_blank_override_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert config.get_api_key("   ") == "sk-from-env"


def test_settings_defaults():
    settings = config.get_settings()
    assert settings.chat_model == config.DEFAULT_CHAT_MODEL
    assert settings.temperature == 0.3


def test_settings_bad_temperature_falls_back(monkeypatch):
    monkeypatch.setenv("OPENAI_TEMPERATURE", "warm")
    assert config.get_settings().temperature == 0.3


def test_mask_key():
    assert config.mask_key("sk-1234567890abcd") == "sk-...abcd"
    assert config.mask_key("short") == "*****"
