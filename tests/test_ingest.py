from pathlib import Path

import pytest

from edupulse.ingest import index_pdf
from edupulse.pdf_parser import PDFParseError
from edupulse.vector_store import VectorStore, document_id

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "cell_biology_notes.pdf"


@pytest.fixture
def store(tmp_path):
    return VectorStore.persistent(str(tmp_path / "chroma"))


def test_index_pdf_end_to_end(store, keyword_embedder):
    data = SAMPLE.read_bytes()
    info, cached = index_pdf(data, "cells.pdf", store, keyword_embedder)
    assert not cached
    assert info.doc_id == document_id(data)
    assert info.page_count == 3
    assert info.chunk_count == len(store.get_chunks(info.doc_id))

    results = store.search(info.doc_id, keyword_embedder.embed_query("Golgi apparatus"), k=1)
    assert "Golgi" in results[0].chunk.text


def test_same_file_is_not_indexed_twice(store, keyword_embedder):
    data = SAMPLE.read_bytes()
    index_pdf(data, "cells.pdf", store, keyword_embedder)
    info, cached = index_pdf(data, "renamed.pdf", store, keyword_embedder)
    assert cached
    assert info.filename == "cells.pdf"
    assert keyword_embedder.calls == 1


def test_progress_is_reported(store, keyword_embedder):
    progress = []
    index_pdf(
        SAMPLE.read_bytes(),
        "cells.pdf",
        store,
        keyword_embedder,
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert progress and progress[-1][0] == progress[-1][1]


def test_steps_are_reported(store, keyword_embedder):
    steps = []
    index_pdf(SAMPLE.read_bytes(), "cells.pdf", store, keyword_embedder, on_step=steps.append)
    assert steps[0] == "Reading the PDF..."
    assert "Found 3 pages." in steps
    assert any(s.startswith("Split the text into") for s in steps)
    assert steps[-1] == "Saving to the vector database..."

    steps.clear()
    index_pdf(SAMPLE.read_bytes(), "cells.pdf", store, keyword_embedder, on_step=steps.append)
    assert steps == ["Already indexed, reusing the saved index."]


def test_bad_pdf_raises_parse_error(store, keyword_embedder):
    with pytest.raises(PDFParseError):
        index_pdf(b"nope", "bad.pdf", store, keyword_embedder)
    assert store.list_documents() == []
