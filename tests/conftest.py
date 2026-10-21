"""Shared test helpers: a fake OpenAI client that never touches the network."""

import math
import re
import zlib
from types import SimpleNamespace

import pytest


class FakeCompletions:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []
        # what .parse() returns: a dict validated against response_format
        self.parsed: dict | None = None
        self.refusal: str | None = None

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        model = kwargs["response_format"]
        parsed = None if self.parsed is None else model.model_validate(self.parsed)
        message = SimpleNamespace(parsed=parsed, refusal=self.refusal, content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

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


class FakeEmbeddings:
    """Returns tiny deterministic vectors so tests can check order and batching."""

    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text)), float(i)])
            for i, text in enumerate(inputs)
        ]
        # the real API doesn't promise order, so hand the items back reversed
        return SimpleNamespace(data=list(reversed(data)))


class FakeOpenAI:
    def __init__(self, reply: str = "fake reply"):
        self.chat = SimpleNamespace(completions=FakeCompletions(reply))
        self.embeddings = FakeEmbeddings()


@pytest.fixture
def fake_client():
    return FakeOpenAI()


class KeywordEmbedder:
    """Bag-of-words "embeddings": texts sharing words end up close together.

    Good enough to test similarity search without calling an API.
    """

    dims = 256

    def __init__(self):
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in re.findall(r"[a-z]+", text.lower()):
            vec[zlib.crc32(word.encode()) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts, on_progress=None):
        self.calls += 1
        vectors = [self._vector(t) for t in texts]
        if on_progress:
            on_progress(len(texts), len(texts))
        return vectors

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def keyword_embedder():
    return KeywordEmbedder()
