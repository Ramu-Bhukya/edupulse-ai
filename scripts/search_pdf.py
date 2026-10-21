"""Index a PDF and print the chunks that best match a question.

Usage:
    python scripts/search_pdf.py samples/cell_biology_notes.pdf "What does the Golgi do?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edupulse.chunker import chunk_document  # noqa: E402
from edupulse.config import MissingAPIKeyError  # noqa: E402
from edupulse.embeddings import OpenAIEmbedder  # noqa: E402
from edupulse.llm import get_client  # noqa: E402
from edupulse.pdf_parser import extract_pages  # noqa: E402
from edupulse.vector_store import VectorStore, document_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", type=Path)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=3, help="number of chunks to show")
    args = parser.parse_args()

    try:
        embedder = OpenAIEmbedder(get_client())
    except MissingAPIKeyError as exc:
        print(exc)
        return 1

    data = args.pdf.read_bytes()
    doc_id = document_id(data)
    store = VectorStore.persistent()

    if store.has_document(doc_id):
        print(f"{args.pdf.name} is already indexed.")
    else:
        doc = extract_pages(data, filename=args.pdf.name)
        chunks = chunk_document(doc)
        print(f"Indexing {doc.page_count} pages as {len(chunks)} chunks...")
        vectors = embedder.embed([c.text for c in chunks])
        store.add_document(doc_id, doc.filename, chunks, vectors, page_count=doc.page_count)

    results = store.search(doc_id, embedder.embed_query(args.question), k=args.k)
    for rank, result in enumerate(results, start=1):
        print(f"\n#{rank}  {result.chunk.page_label}  similarity={result.similarity:.3f}")
        print(result.chunk.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
