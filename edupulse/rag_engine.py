"""Retrieval-augmented generation (RAG) over an uploaded document.

question -> embed -> top-k chunks from Chroma -> chunks go into the
system prompt as context -> the model answers from that context.
"""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from edupulse.embeddings import Embedder
from edupulse.llm import stream_chat
from edupulse.prompts import IDK_ANSWER, RAG_RULES, build_chat_messages
from edupulse.vector_store import SearchResult, VectorStore

DEFAULT_TOP_K = 4
SHORT_FOLLOW_UP_WORDS = 6


def retrieval_query(history: list[dict]) -> str:
    """Text to search the document with, based on the latest user message.

    Follow-ups like "why?" or "explain that again" are useless as search
    queries on their own, so short ones are combined with the previous
    user question.
    """
    user_messages = [m["content"] for m in history if m["role"] == "user"]
    if not user_messages:
        return ""
    latest = user_messages[-1]
    if len(latest.split()) < SHORT_FOLLOW_UP_WORDS and len(user_messages) > 1:
        return f"{user_messages[-2]} {latest}"
    return latest


def retrieve(
    store: VectorStore, embedder: Embedder, doc_id: str, query: str, k: int = DEFAULT_TOP_K
) -> list[SearchResult]:
    if not query.strip():
        return []
    return store.search(doc_id, embedder.embed_query(query), k=k)


def format_context(results: list[SearchResult]) -> str:
    """Numbered context block, e.g. "[1] (p. 2)\\n<chunk text>"."""
    return "\n\n".join(
        f"[{i}] ({r.chunk.page_label})\n{r.chunk.text}" for i, r in enumerate(results, start=1)
    )


def build_rag_system_prompt(base_prompt: str, context: str) -> str:
    """Persona prompt + grounding rules + the retrieved excerpts.

    The excerpts sit inside <context> tags so the model can tell the
    document apart from our instructions.
    """
    return f"{base_prompt}\n\n{RAG_RULES}\n\n<context>\n{context}\n</context>"


def is_idk(answer: str) -> bool:
    """True if the model said it couldn't find the answer in the document."""
    return answer.strip().lower().startswith(IDK_ANSWER.lower().rstrip("."))


def answer_stream(
    client: OpenAI,
    store: VectorStore,
    embedder: Embedder,
    doc_id: str,
    base_prompt: str,
    history: list[dict],
    *,
    k: int = DEFAULT_TOP_K,
    max_messages: int = 10,
) -> tuple[list[SearchResult], Iterator[str]]:
    """Retrieve context for the latest question and stream the answer.

    Returns the retrieved chunks (to show as sources) and a token stream.
    """
    results = retrieve(store, embedder, doc_id, retrieval_query(history), k=k)
    if not results:
        # nothing to ground the answer in, so don't even ask the model
        return [], iter([IDK_ANSWER])
    system_prompt = build_rag_system_prompt(base_prompt, format_context(results))
    messages = build_chat_messages(system_prompt, history, max_messages=max_messages)
    return results, stream_chat(client, messages)
