"""Store document chunks and their embeddings in ChromaDB.

Each uploaded PDF gets its own collection named after a hash of the
file, so uploading the same file twice can reuse the existing index.
We pass our own OpenAI vectors in, so Chroma never needs to download
its default embedding model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from edupulse.chunker import Chunk

DEFAULT_PERSIST_DIR = ".chroma"
COLLECTION_PREFIX = "doc_"


def document_id(data: bytes) -> str:
    """Stable id for a file: the first 16 hex chars of its SHA-256."""
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    distance: float  # cosine distance: 0 = same direction, 2 = opposite

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int


class VectorStore:
    def __init__(self, client: chromadb.ClientAPI):
        self.client = client

    @classmethod
    def persistent(cls, path: str = DEFAULT_PERSIST_DIR) -> VectorStore:
        settings = ChromaSettings(anonymized_telemetry=False)
        return cls(chromadb.PersistentClient(path=path, settings=settings))

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _collection_name(doc_id: str) -> str:
        return f"{COLLECTION_PREFIX}{doc_id}"

    def _collection(self, doc_id: str):
        return self.client.get_collection(self._collection_name(doc_id), embedding_function=None)

    @staticmethod
    def _info(doc_id: str, metadata: dict | None) -> DocumentInfo:
        metadata = metadata or {}
        return DocumentInfo(
            doc_id=doc_id,
            filename=str(metadata.get("filename", "unknown.pdf")),
            page_count=int(metadata.get("page_count", 0)),
            chunk_count=int(metadata.get("chunk_count", 0)),
        )

    # --- documents ------------------------------------------------------------

    def has_document(self, doc_id: str) -> bool:
        return self.get_document(doc_id) is not None

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        try:
            collection = self._collection(doc_id)
        except NotFoundError:
            return None
        return self._info(doc_id, collection.metadata)

    def list_documents(self) -> list[DocumentInfo]:
        docs = []
        for collection in self.client.list_collections():
            if collection.name.startswith(COLLECTION_PREFIX):
                doc_id = collection.name[len(COLLECTION_PREFIX) :]
                docs.append(self._info(doc_id, collection.metadata))
        return sorted(docs, key=lambda d: d.filename.lower())

    def add_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        page_count: int,
    ) -> DocumentInfo:
        """Index a document's chunks. Replaces any existing index for doc_id."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            raise ValueError("Nothing to index: the document produced no chunks")

        self.delete_document(doc_id)
        collection = self.client.create_collection(
            self._collection_name(doc_id),
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={
                "filename": filename,
                "page_count": page_count,
                "chunk_count": len(chunks),
            },
        )

        batch = self.client.get_max_batch_size()
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            collection.add(
                ids=[f"{doc_id}-{c.index}" for c in part],
                documents=[c.text for c in part],
                embeddings=embeddings[start : start + batch],
                metadatas=[
                    {"index": c.index, "page": c.page, "page_end": c.page_end} for c in part
                ],
            )
        return DocumentInfo(doc_id, filename, page_count, len(chunks))

    def delete_document(self, doc_id: str) -> None:
        try:
            self.client.delete_collection(self._collection_name(doc_id))
        except NotFoundError:
            pass

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """All chunks of a document in their original order."""
        result = self._collection(doc_id).get(include=["documents", "metadatas"])
        chunks = [
            _to_chunk(text, meta)
            for text, meta in zip(result["documents"], result["metadatas"], strict=True)
        ]
        return sorted(chunks, key=lambda c: c.index)

    # --- search ---------------------------------------------------------------

    def search(self, doc_id: str, query_embedding: list[float], k: int = 3) -> list[SearchResult]:
        """Return the k chunks closest to the query, best match first."""
        collection = self._collection(doc_id)
        count = collection.count()
        if count == 0:
            return []
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, min(k, count)),
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(chunk=_to_chunk(text, meta), distance=float(distance))
            for text, meta, distance in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]


def _to_chunk(text: str, meta: dict) -> Chunk:
    return Chunk(
        text=text,
        index=int(meta["index"]),
        page=int(meta["page"]),
        page_end=int(meta["page_end"]),
    )
