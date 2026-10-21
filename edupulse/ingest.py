"""PDF -> text -> chunks -> embeddings -> vector store, in one call."""

from __future__ import annotations

from edupulse.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_document
from edupulse.embeddings import Embedder, ProgressCallback
from edupulse.pdf_parser import extract_pages
from edupulse.vector_store import DocumentInfo, VectorStore, document_id


def index_pdf(
    data: bytes,
    filename: str,
    store: VectorStore,
    embedder: Embedder,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    on_progress: ProgressCallback | None = None,
) -> tuple[DocumentInfo, bool]:
    """Index a PDF unless the exact same file is already in the store.

    Returns the document info and whether it was already indexed.
    """
    doc_id = document_id(data)
    existing = store.get_document(doc_id)
    if existing is not None:
        return existing, True

    doc = extract_pages(data, filename=filename)
    chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
    vectors = embedder.embed([c.text for c in chunks], on_progress=on_progress)
    info = store.add_document(doc_id, doc.filename, chunks, vectors, page_count=doc.page_count)
    return info, False
