import pytest

from edupulse.chunker import Chunk
from edupulse.vector_store import VectorStore, document_id

CHUNKS = [
    Chunk("Mitochondria produce ATP through aerobic respiration.", 0, 1, 1),
    Chunk("The nucleus stores DNA and controls gene expression.", 1, 1, 2),
    Chunk("Osmosis is the diffusion of water across a membrane.", 2, 2, 2),
]


@pytest.fixture
def store(tmp_path):
    return VectorStore.persistent(str(tmp_path / "chroma"))


def index_sample(store, embedder, doc_id="abc123"):
    vectors = embedder.embed([c.text for c in CHUNKS])
    return store.add_document(doc_id, "cells.pdf", CHUNKS, vectors, page_count=2)


def test_document_id_is_stable_and_short():
    assert document_id(b"hello") == document_id(b"hello")
    assert document_id(b"hello") != document_id(b"world")
    assert len(document_id(b"hello")) == 16


def test_add_and_describe_document(store, keyword_embedder):
    info = index_sample(store, keyword_embedder)
    assert info.chunk_count == 3
    assert store.has_document("abc123")
    assert store.get_document("abc123") == info
    assert store.list_documents() == [info]


def test_unknown_document(store):
    assert store.get_document("missing") is None
    assert not store.has_document("missing")
    store.delete_document("missing")  # no error


def test_get_chunks_round_trip(store, keyword_embedder):
    index_sample(store, keyword_embedder)
    assert store.get_chunks("abc123") == CHUNKS


def test_reindexing_replaces_old_chunks(store, keyword_embedder):
    index_sample(store, keyword_embedder)
    one = [Chunk("Only chunk now.", 0, 1, 1)]
    store.add_document("abc123", "cells.pdf", one, keyword_embedder.embed(["x"]), page_count=1)
    assert store.get_chunks("abc123") == one
    assert store.get_document("abc123").chunk_count == 1


def test_delete_document(store, keyword_embedder):
    index_sample(store, keyword_embedder)
    store.delete_document("abc123")
    assert store.list_documents() == []


def test_mismatched_embeddings_rejected(store):
    with pytest.raises(ValueError):
        store.add_document("x", "x.pdf", CHUNKS, [[0.1, 0.2]], page_count=1)


def test_empty_document_rejected(store):
    with pytest.raises(ValueError):
        store.add_document("x", "x.pdf", [], [], page_count=1)


def test_index_survives_reopening(tmp_path, keyword_embedder):
    path = str(tmp_path / "chroma")
    index_sample(VectorStore.persistent(path), keyword_embedder)
    reopened = VectorStore.persistent(path)
    assert reopened.get_document("abc123").filename == "cells.pdf"
