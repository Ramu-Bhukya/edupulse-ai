import pytest

from edupulse.chunker import Chunk
from edupulse.rag_engine import (
    answer_stream,
    build_rag_system_prompt,
    format_context,
    retrieval_query,
    retrieve,
)
from edupulse.vector_store import SearchResult, VectorStore

CHUNKS = [
    Chunk("Mitochondria produce ATP through aerobic respiration.", 0, 1, 1),
    Chunk("The nucleus stores DNA and controls gene expression.", 1, 2, 2),
    Chunk("Osmosis is the diffusion of water across a membrane.", 2, 3, 3),
]


@pytest.fixture
def store(tmp_path, keyword_embedder):
    store = VectorStore.persistent(str(tmp_path / "chroma"))
    vectors = keyword_embedder.embed([c.text for c in CHUNKS])
    store.add_document("doc1", "cells.pdf", CHUNKS, vectors, page_count=3)
    return store


def test_retrieval_query_uses_latest_question():
    history = [
        {"role": "user", "content": "What do mitochondria produce for the cell?"},
        {"role": "assistant", "content": "ATP."},
        {"role": "user", "content": "How does osmosis move water across membranes?"},
    ]
    assert retrieval_query(history) == "How does osmosis move water across membranes?"


def test_short_follow_up_is_combined_with_previous_question():
    history = [
        {"role": "user", "content": "What do mitochondria produce?"},
        {"role": "assistant", "content": "ATP."},
        {"role": "user", "content": "Why?"},
    ]
    assert retrieval_query(history) == "What do mitochondria produce? Why?"


def test_retrieval_query_empty_history():
    assert retrieval_query([]) == ""


def test_retrieve(store, keyword_embedder):
    results = retrieve(store, keyword_embedder, "doc1", "Where is DNA stored?", k=2)
    assert len(results) == 2
    assert "nucleus" in results[0].chunk.text
    assert retrieve(store, keyword_embedder, "doc1", "   ") == []


def test_format_context_numbers_chunks_with_pages():
    results = [SearchResult(CHUNKS[1], 0.1), SearchResult(CHUNKS[0], 0.4)]
    context = format_context(results)
    assert context.startswith("[1] (p. 2)\nThe nucleus")
    assert "[2] (p. 1)\nMitochondria" in context


def test_rag_system_prompt_contains_context():
    prompt = build_rag_system_prompt("You are a tutor.", "[1] (p. 1)\nSome text")
    assert prompt.startswith("You are a tutor.")
    assert "Context:\n[1] (p. 1)\nSome text" in prompt


def test_answer_stream_sends_context_and_history(store, keyword_embedder, fake_client):
    history = [{"role": "user", "content": "How does osmosis move water?"}]
    results, stream = answer_stream(
        fake_client, store, keyword_embedder, "doc1", "You are a tutor.", history, k=1
    )
    assert "Osmosis" in results[0].chunk.text
    assert "".join(stream) == "fake reply"

    sent = fake_client.chat.completions.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "Osmosis is the diffusion of water" in sent[0]["content"]
    assert sent[-1] == history[-1]
