import pytest
from openai import OpenAI
from pydantic import BaseModel

from edupulse.config import MissingAPIKeyError
from edupulse.llm import (
    StructuredOutputError,
    complete,
    get_client,
    parse_structured,
    stream_chat,
)


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


class Answer(BaseModel):
    value: int


def test_parse_structured_returns_model(fake_client):
    fake_client.chat.completions.parsed = {"value": 42}
    result = parse_structured(fake_client, [{"role": "user", "content": "6*7"}], Answer)
    assert result == Answer(value=42)
    assert fake_client.chat.completions.calls[0]["response_format"] is Answer


def test_parse_structured_refusal(fake_client):
    fake_client.chat.completions.refusal = "I can't help with that."
    with pytest.raises(StructuredOutputError, match="refused"):
        parse_structured(fake_client, [{"role": "user", "content": "x"}], Answer)


def test_parse_structured_missing_data(fake_client):
    with pytest.raises(StructuredOutputError):
        parse_structured(fake_client, [{"role": "user", "content": "x"}], Answer)
