"""Extract text from PDF files with pypdf."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from pypdf import PasswordType, PdfReader
from pypdf.errors import DependencyError, PdfReadError


class PDFParseError(ValueError):
    """The PDF could not be read or contains no extractable text."""


@dataclass
class Page:
    number: int  # 1-based, as shown in a PDF viewer
    text: str


@dataclass
class ParsedDocument:
    filename: str
    pages: list[Page] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def empty_pages(self) -> list[int]:
        """Pages with no text, usually scanned images or blank pages."""
        return [p.number for p in self.pages if not p.text.strip()]


# --- cleaning -----------------------------------------------------------------

_HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_PAGE_NUMBER = re.compile(r"^[\s\-–—]*(page\s*)?\d+(\s*(of|/)\s*\d+)?[\s\-–—]*$", re.I)
EDGE_LINES = 2  # how many lines at the top/bottom of a page count as header/footer


def clean_text(text: str) -> str:
    """Tidy up raw text from pypdf without changing its meaning."""
    text = unicodedata.normalize("NFKC", text)  # e.g. the "ﬁ" ligature -> "fi"
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1", text)  # "trans-\nport" -> "transport"
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _BLANK_LINES.sub("\n\n", text).strip()


def _line_key(line: str) -> str:
    # "Page 3" and "Page 14" should count as the same footer
    return re.sub(r"\d+", "#", line.strip().lower())


def _edge_indexes(n_lines: int) -> set[int]:
    # On short pages only the very first/last line can be a header/footer,
    # otherwise every line would count as an "edge".
    k = EDGE_LINES if n_lines > 2 * EDGE_LINES else 1
    return set(range(min(k, n_lines))) | set(range(max(n_lines - k, 0), n_lines))


def remove_repeated_edge_lines(pages: list[str], min_share: float = 0.6) -> list[str]:
    """Drop running headers/footers and bare page numbers.

    A line near the top or bottom of a page is treated as a header/footer
    if (after replacing digits) it shows up on at least `min_share` of the
    pages. Needs 3+ pages to say anything useful.
    """
    split = [page.split("\n") for page in pages]
    repeated: set[str] = set()
    if len(pages) >= 3:
        counts: Counter[str] = Counter()
        for lines in split:
            edges = {_line_key(lines[i]) for i in _edge_indexes(len(lines)) if lines[i].strip()}
            counts.update(edges)
        threshold = max(2, math.ceil(min_share * len(pages)))
        repeated = {key for key, n in counts.items() if n >= threshold}

    cleaned = []
    for lines in split:
        edges = _edge_indexes(len(lines))
        keep = [
            line
            for i, line in enumerate(lines)
            if not (i in edges and (_line_key(line) in repeated or _PAGE_NUMBER.match(line)))
        ]
        cleaned.append("\n".join(keep))
    return cleaned


# --- reading ------------------------------------------------------------------


def _open_reader(source: bytes | str | Path | BinaryIO) -> PdfReader:
    stream = BytesIO(source) if isinstance(source, bytes) else source
    try:
        reader = PdfReader(stream)
    except (PdfReadError, OSError) as exc:
        raise PDFParseError(f"This file doesn't look like a valid PDF ({exc}).") from exc

    if reader.is_encrypted:
        # Many textbooks are "encrypted" only to stop editing/printing and
        # open fine with an empty password.
        try:
            result = reader.decrypt("")
        except (DependencyError, PdfReadError) as exc:
            raise PDFParseError(f"Couldn't decrypt this PDF ({exc}).") from exc
        if result == PasswordType.NOT_DECRYPTED:
            raise PDFParseError("This PDF is password protected. Remove the password and retry.")
    return reader


def extract_pages(
    source: bytes | str | Path | BinaryIO,
    filename: str | None = None,
    *,
    clean: bool = True,
) -> ParsedDocument:
    """Read every page of a PDF and return its (cleaned) text page by page."""
    if filename is None:
        filename = Path(source).name if isinstance(source, str | Path) else "document.pdf"

    reader = _open_reader(source)
    raw_pages = []
    for page in reader.pages:
        try:
            raw_pages.append(page.extract_text() or "")
        except Exception:  # one broken page shouldn't stop the whole document
            raw_pages.append("")

    if clean:
        raw_pages = [clean_text(t) for t in remove_repeated_edge_lines(raw_pages)]

    doc = ParsedDocument(filename=filename)
    for number, text in enumerate(raw_pages, start=1):
        doc.pages.append(Page(number=number, text=text))

    if not doc.pages:
        raise PDFParseError("This PDF has no pages.")
    if not doc.text.strip():
        raise PDFParseError(
            "No text could be extracted. The PDF is probably scanned images; "
            "OCR is not supported yet."
        )
    return doc
