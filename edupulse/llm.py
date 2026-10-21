"""Thin wrapper around the OpenAI SDK.

Functions take the client as an argument so they are easy to test with
a fake client and so the Streamlit app can cache one client per key.
"""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from edupulse.config import get_api_key, get_settings


def get_client(api_key: str | None = None) -> OpenAI:
    return OpenAI(api_key=get_api_key(api_key))


def complete(
    client: OpenAI,
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Send a list of chat messages and return the reply text."""
    settings = get_settings()
    response = client.chat.completions.create(
        model=model or settings.chat_model,
        messages=messages,
        temperature=settings.temperature if temperature is None else temperature,
    )
    return response.choices[0].message.content or ""


def stream_chat(
    client: OpenAI,
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> Iterator[str]:
    """Yield the reply token by token (works directly with st.write_stream)."""
    settings = get_settings()
    stream = client.chat.completions.create(
        model=model or settings.chat_model,
        messages=messages,
        temperature=settings.temperature if temperature is None else temperature,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
