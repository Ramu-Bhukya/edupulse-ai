"""Turn text into embedding vectors with the OpenAI embeddings API."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol

from openai import OpenAI

from edupulse.config import get_settings

BATCH_SIZE = 100  # well below the API limit, keeps each request small
QUERY_CACHE_SIZE = 256

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
        # Searches, quiz topics and chat follow-ups often repeat the same text.
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()  # Streamlit sessions run in threads

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
        key = text.strip()
        with self._lock:
            if key in self._query_cache:
                self._query_cache.move_to_end(key)
                return self._query_cache[key]
        vector = self.embed([text])[0]
        with self._lock:
            self._query_cache[key] = vector
            if len(self._query_cache) > QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return vector
