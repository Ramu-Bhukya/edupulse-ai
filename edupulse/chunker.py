"""Split document text into overlapping chunks for embedding.

LLMs (and embedding models) have a limited context window, so a long
textbook has to be cut into smaller pieces. A sliding window with some
overlap keeps sentences that sit on a boundary readable in at least one
chunk.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

from edupulse.pdf_parser import ParsedDocument

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
PAGE_SEPARATOR = "\n\n"

# end of a sentence, but not numbered headings/lists like "5. Cell Division"
_SENTENCE_END = re.compile(r"(?<!\b\d)(?<!\b\d\d)[.!?][\"')\]]?\s")


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int  # position of the chunk in the document
    page: int  # page the chunk starts on (1-based)
    page_end: int  # page the chunk ends on

    @property
    def page_label(self) -> str:
        if self.page == self.page_end:
            return f"p. {self.page}"
        return f"pp. {self.page}-{self.page_end}"


def _find_break(text: str, start: int, end: int, min_end: int) -> int:
    """Best place to end a chunk inside text[min_end:end].

    Prefers a paragraph break, then the end of a sentence, then any
    whitespace. Falls back to a hard cut at `end`.
    """
    window = text[min_end:end]
    para = window.rfind("\n\n")
    if para != -1:
        return min_end + para
    sentences = list(_SENTENCE_END.finditer(window))
    if sentences:
        return min_end + sentences[-1].end()
    space = max(window.rfind(" "), window.rfind("\n"))
    if space != -1:
        return min_end + space
    return end


def split_text(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[tuple[int, int]]:
    """Return (start, end) character offsets of each chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    spans: list[tuple[int, int]] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _find_break(text, start, end, min_end=start + chunk_size // 2)

        # trim surrounding whitespace so offsets point at real text
        s, e = start, end
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            spans.append((s, e))

        if end >= n:
            break

        # step back by `overlap`, then move forward to the next sentence start
        # (or at least a word boundary) so chunks don't begin mid-sentence
        next_start = max(end - overlap, start + 1)
        sentence = _SENTENCE_END.search(text, next_start, end)
        if sentence and sentence.end() < end:
            next_start = sentence.end()
        else:
            while 0 < next_start < end and not text[next_start - 1].isspace():
                next_start += 1
        start = next_start if next_start < end else end
    return spans


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk a parsed PDF and remember which page(s) each chunk came from."""
    parts, page_starts, page_numbers = [], [], []
    offset = 0
    for page in doc.pages:
        if not page.text.strip():
            continue
        page_starts.append(offset)
        page_numbers.append(page.number)
        parts.append(page.text)
        offset += len(page.text) + len(PAGE_SEPARATOR)
    full_text = PAGE_SEPARATOR.join(parts)

    def page_at(pos: int) -> int:
        return page_numbers[bisect.bisect_right(page_starts, pos) - 1]

    chunks = []
    for i, (start, end) in enumerate(split_text(full_text, chunk_size, overlap)):
        chunks.append(
            Chunk(
                text=full_text[start:end],
                index=i,
                page=page_at(start),
                page_end=page_at(end - 1),
            )
        )
    return chunks
