"""Thin wrapper around the OpenAI SDK.

Functions take the client as an argument so they are easy to test with
a fake client and so the Streamlit app can cache one client per key.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel

from edupulse.config import MissingAPIKeyError, get_api_key, get_settings

ModelT = TypeVar("ModelT", bound=BaseModel)


def get_client(api_key: str | None = None) -> OpenAI:
    return OpenAI(api_key=get_api_key(api_key), max_retries=2, timeout=60)


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


class StructuredOutputError(RuntimeError):
    """The model refused or returned something that didn't match the schema."""


def parse_structured(
    client: OpenAI,
    messages: list[dict],
    response_model: type[ModelT],
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> ModelT:
    """Ask the model for JSON matching a Pydantic model and return the parsed object.

    Uses OpenAI structured outputs, so the reply is guaranteed to follow
    the JSON schema generated from `response_model`.
    """
    settings = get_settings()
    completion = client.chat.completions.parse(
        model=model or settings.chat_model,
        messages=messages,
        response_format=response_model,
        temperature=settings.temperature if temperature is None else temperature,
    )
    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise StructuredOutputError(f"The model refused: {message.refusal}")
    if message.parsed is None:
        raise StructuredOutputError("The model did not return valid structured data.")
    return message.parsed


# Everything the UI should catch and show as a message instead of a traceback.
LLM_ERRORS = (MissingAPIKeyError, StructuredOutputError, openai.OpenAIError)


def friendly_error(exc: Exception) -> str:
    """Turn SDK exceptions into a message a student can act on."""
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check that it is correct and still active."
    if isinstance(exc, openai.RateLimitError):
        if getattr(exc, "code", None) == "insufficient_quota":
            return "This API key has run out of credit. Check your OpenAI billing settings."
        return "Too many requests right now. Wait a few seconds and try again."
    if isinstance(exc, openai.NotFoundError):
        return (
            f"The model '{get_settings().chat_model}' was not found. "
            "Set OPENAI_MODEL to a model your key can use."
        )
    if isinstance(exc, openai.APITimeoutError):
        return "OpenAI took too long to answer. Please try again."
    if isinstance(exc, openai.APIConnectionError):
        return "Couldn't reach OpenAI. Check your internet connection."
    if isinstance(exc, openai.BadRequestError):
        return f"OpenAI couldn't process the request: {exc.message}"
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI returned an error ({exc.status_code}). Please try again later."
    return str(exc) or exc.__class__.__name__
