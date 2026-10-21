from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from edupulse.pdf_parser import PDFParseError, extract_pages

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "cell_biology_notes.pdf"


def _to_bytes(writer: PdfWriter) -> bytes:
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_extracts_every_page_from_sample():
    doc = extract_pages(SAMPLE)
    assert doc.filename == "cell_biology_notes.pdf"
    assert doc.page_count >= 3
    assert [p.number for p in doc.pages] == list(range(1, doc.page_count + 1))
    assert "Cell theory" in doc.pages[0].text
    assert "sodium-potassium pump" in doc.text
    assert doc.empty_pages == []


def test_accepts_bytes_and_file_objects():
    data = SAMPLE.read_bytes()
    from_bytes = extract_pages(data, filename="notes.pdf")
    with open(SAMPLE, "rb") as f:
        from_file = extract_pages(f, filename="notes.pdf")
    assert from_bytes.filename == "notes.pdf"
    assert from_bytes.text == from_file.text


def test_not_a_pdf():
    with pytest.raises(PDFParseError, match="valid PDF"):
        extract_pages(b"definitely not a pdf", filename="fake.pdf")


def test_blank_pdf_has_no_text():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with pytest.raises(PDFParseError, match="No text"):
        extract_pages(_to_bytes(writer))


def test_password_protected_pdf():
    writer = PdfWriter(clone_from=PdfReader(SAMPLE))
    writer.encrypt(user_password="secret", algorithm="AES-256")
    with pytest.raises(PDFParseError, match="password"):
        extract_pages(_to_bytes(writer))


def test_owner_only_encryption_still_opens():
    writer = PdfWriter(clone_from=PdfReader(SAMPLE))
    writer.encrypt(user_password="", owner_password="owner", algorithm="AES-256")
    doc = extract_pages(_to_bytes(writer))
    assert "Cell theory" in doc.text


def test_blank_pages_are_reported():
    writer = PdfWriter(clone_from=PdfReader(SAMPLE))
    writer.add_blank_page()
    doc = extract_pages(_to_bytes(writer))
    assert doc.empty_pages == [doc.page_count]
