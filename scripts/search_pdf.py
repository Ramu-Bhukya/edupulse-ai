"""Index a PDF and print the chunks that best match a question.

Usage:
    python scripts/search_pdf.py samples/cell_biology_notes.pdf "What does the Golgi do?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edupulse.config import MissingAPIKeyError, get_settings  # noqa: E402
from edupulse.embeddings import OpenAIEmbedder  # noqa: E402
from edupulse.ingest import index_pdf  # noqa: E402
from edupulse.llm import get_client  # noqa: E402
from edupulse.vector_store import VectorStore  # noqa: E402


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

    store = VectorStore.persistent(get_settings().chroma_dir)
    info, cached = index_pdf(args.pdf.read_bytes(), args.pdf.name, store, embedder)
    status = "already indexed" if cached else "indexed"
    print(f"{info.filename}: {status} ({info.page_count} pages, {info.chunk_count} chunks)")

    results = store.search(info.doc_id, embedder.embed_query(args.question), k=args.k)
    for rank, result in enumerate(results, start=1):
        print(f"\n#{rank}  {result.chunk.page_label}  similarity={result.similarity:.3f}")
        print(result.chunk.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
