"""Shared test helpers: a fake OpenAI client that never touches the network."""

from types import SimpleNamespace

import pytest


class FakeCompletions:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    def __init__(self, reply: str = "fake reply"):
        self.chat = SimpleNamespace(completions=FakeCompletions(reply))


@pytest.fixture
def fake_client():
    return FakeOpenAI()
