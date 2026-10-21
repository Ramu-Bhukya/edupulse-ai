"""Small text helpers for displaying document snippets."""

from __future__ import annotations

import re

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#+|$~])")

STOPWORDS = {
    "about", "after", "also", "been", "does", "from", "have", "into", "more", "most",
    "only", "other", "some", "than", "that", "their", "them", "then", "there", "these",
    "they", "this", "what", "when", "where", "which", "while", "with", "would", "your",
}  # fmt: skip


def escape_markdown(text: str) -> str:
    """Escape characters that st.markdown would treat as formatting (or LaTeX)."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def query_terms(query: str, min_length: int = 4) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", query.lower())
    terms = {w for w in words if len(w) >= min_length and w not in STOPWORDS}
    return sorted(terms, key=len, reverse=True)


def highlight_terms(text: str, query: str) -> str:
    """Escape `text` for Markdown and bold the words that appear in `query`.

    Matches word starts, so "cell" also highlights "cells" and "cellular".
    """
    safe = escape_markdown(text)
    terms = query_terms(query)
    if not terms:
        return safe
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\w*", re.I)
    return pattern.sub(lambda m: f"**{m.group(0)}**", safe)
