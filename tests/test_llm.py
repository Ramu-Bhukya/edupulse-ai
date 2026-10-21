import pytest
from openai import OpenAI

from edupulse.config import MissingAPIKeyError
from edupulse.llm import complete, get_client, stream_chat


def test_complete_returns_reply_text(fake_client):
    messages = [{"role": "user", "content": "hi"}]
    assert complete(fake_client, messages) == "fake reply"
    call = fake_client.chat.completions.calls[0]
    assert call["messages"] == messages
    assert call["model"]


def test_complete_model_override(fake_client):
    complete(fake_client, [{"role": "user", "content": "hi"}], model="my-model", temperature=0)
    call = fake_client.chat.completions.calls[0]
    assert call["model"] == "my-model"
    assert call["temperature"] == 0


def test_get_client_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        get_client()


def test_get_client_with_key():
    assert isinstance(get_client("sk-test"), OpenAI)


def test_stream_chat_yields_tokens(fake_client):
    tokens = list(stream_chat(fake_client, [{"role": "user", "content": "hi"}]))
    assert tokens == ["fake ", "reply"]
    assert "".join(tokens) == "fake reply"
    assert fake_client.chat.completions.calls[0]["stream"] is True
