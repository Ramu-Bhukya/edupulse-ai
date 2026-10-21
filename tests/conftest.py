"""Shared test helpers: a fake OpenAI client that never touches the network."""

from types import SimpleNamespace

import pytest


class FakeCompletions:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self._stream()
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def _stream(self):
        # Mimic the SDK: one chunk per word, then a final chunk with no content.
        words = self.reply.split(" ")
        for i, word in enumerate(words):
            text = word if i == len(words) - 1 else word + " "
            delta = SimpleNamespace(content=text)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])


class FakeOpenAI:
    def __init__(self, reply: str = "fake reply"):
        self.chat = SimpleNamespace(completions=FakeCompletions(reply))


@pytest.fixture
def fake_client():
    return FakeOpenAI()
