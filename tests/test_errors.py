import httpx2
import openai
import pytest

from edupulse.config import MissingAPIKeyError
from edupulse.llm import StructuredOutputError, friendly_error

REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def status_error(cls, status, body=None):
    response = httpx2.Response(status, request=REQUEST)
    return cls("boom", response=response, body=body)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (status_error(openai.AuthenticationError, 401), "rejected the API key"),
        (status_error(openai.RateLimitError, 429), "Too many requests"),
        (
            status_error(openai.RateLimitError, 429, {"code": "insufficient_quota"}),
            "run out of credit",
        ),
        (status_error(openai.NotFoundError, 404), "was not found"),
        (status_error(openai.BadRequestError, 400), "couldn't process"),
        (status_error(openai.InternalServerError, 500), "(500)"),
        (openai.APIConnectionError(request=REQUEST), "Couldn't reach OpenAI"),
        (openai.APITimeoutError(request=REQUEST), "too long"),
        (MissingAPIKeyError("No OpenAI API key found."), "No OpenAI API key found."),
        (StructuredOutputError("The model refused: no"), "refused"),
    ],
)
def test_friendly_error_messages(exc, expected):
    assert expected in friendly_error(exc)
