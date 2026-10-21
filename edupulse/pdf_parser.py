"""Extract text from PDF files with pypdf."""

from __future__ import annotations

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
    source: bytes | str | Path | BinaryIO, filename: str | None = None
) -> ParsedDocument:
    """Read every page of a PDF and return its text page by page."""
    if filename is None:
        filename = Path(source).name if isinstance(source, str | Path) else "document.pdf"

    reader = _open_reader(source)
    doc = ParsedDocument(filename=filename)
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # one broken page shouldn't stop the whole document
            text = ""
        doc.pages.append(Page(number=number, text=text))

    if not doc.pages:
        raise PDFParseError("This PDF has no pages.")
    if not doc.text.strip():
        raise PDFParseError(
            "No text could be extracted. The PDF is probably scanned images; "
            "OCR is not supported yet."
        )
    return doc
