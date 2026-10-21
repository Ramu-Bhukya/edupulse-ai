"""Turn text into embedding vectors with the OpenAI embeddings API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from openai import OpenAI

from edupulse.config import get_settings

BATCH_SIZE = 100  # well below the API limit, keeps each request small

ProgressCallback = Callable[[int, int], None]  # (done, total)


class Embedder(Protocol):
    """Anything that can embed documents and queries (real or fake)."""

    def embed(
        self, texts: list[str], on_progress: ProgressCallback | None = None
    ) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    def __init__(self, client: OpenAI, model: str | None = None, batch_size: int = BATCH_SIZE):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.client = client
        self.model = model or get_settings().embed_model
        self.batch_size = batch_size

    def embed(
        self, texts: list[str], on_progress: ProgressCallback | None = None
    ) -> list[list[float]]:
        """Embed texts in batches, keeping the input order."""
        vectors: list[list[float]] = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            # the API rejects empty strings
            batch = [t if t.strip() else " " for t in texts[i : i + self.batch_size]]
            response = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
            if on_progress:
                on_progress(min(i + self.batch_size, total), total)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
