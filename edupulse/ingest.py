"""PDF -> text -> chunks -> embeddings -> vector store, in one call."""

from __future__ import annotations

from collections.abc import Callable

from edupulse.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_document
from edupulse.embeddings import Embedder, ProgressCallback
from edupulse.pdf_parser import extract_pages
from edupulse.vector_store import DocumentInfo, VectorStore, document_id

StepCallback = Callable[[str], None]


def index_pdf(
    data: bytes,
    filename: str,
    store: VectorStore,
    embedder: Embedder,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    on_step: StepCallback | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[DocumentInfo, bool]:
    """Index a PDF unless the exact same file is already in the store.

    `on_step` receives short status messages for the UI and `on_progress`
    is passed to the embedder. Returns the document info and whether it
    was already indexed.
    """
    step = on_step or (lambda message: None)

    doc_id = document_id(data)
    existing = store.get_document(doc_id)
    if existing is not None:
        step("Already indexed, reusing the saved index.")
        return existing, True

    step("Reading the PDF...")
    doc = extract_pages(data, filename=filename)
    found = f"Found {doc.page_count} pages"
    if doc.empty_pages:
        found += f" ({len(doc.empty_pages)} without text, maybe scanned images)"
    step(found + ".")

    chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
    step(f"Split the text into {len(chunks)} chunks.")

    step("Creating embeddings...")
    vectors = embedder.embed([c.text for c in chunks], on_progress=on_progress)

    step("Saving to the vector database...")
    info = store.add_document(doc_id, doc.filename, chunks, vectors, page_count=doc.page_count)
    return info, False
