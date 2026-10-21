from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from edupulse.pdf_parser import (
    PDFParseError,
    clean_text,
    extract_pages,
    remove_repeated_edge_lines,
)

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


# --- cleaning -----------------------------------------------------------------


def test_clean_text_fixes_ligatures_hyphens_and_spacing():
    raw = "The ﬁrst   step of active trans-\nport\t uses ATP.\n\n\n\nNext   line "
    assert clean_text(raw) == "The first step of active transport uses ATP.\n\nNext line"


def test_clean_text_keeps_real_hyphens():
    # a capitalised continuation is probably a real hyphen/name, leave it
    assert clean_text("Sodium-\nPotassium") == "Sodium-\nPotassium"
    assert clean_text("well-known") == "well-known"


def test_remove_repeated_headers_and_footers():
    pages = [
        f"My Textbook - Chapter 1\nBody text of page {n}.\nMore text.\nPage {n}"
        for n in range(1, 5)
    ]
    cleaned = remove_repeated_edge_lines(pages)
    for n, page in enumerate(cleaned, start=1):
        assert "My Textbook" not in page
        assert "Page" not in page
        assert f"Body text of page {n}." in page


def test_bare_page_numbers_removed_even_on_short_docs():
    cleaned = remove_repeated_edge_lines(["Intro text\n1", "- 2 -\nMore text"])
    assert cleaned == ["Intro text", "More text"]


def test_short_documents_keep_their_first_lines():
    # with only two pages we can't tell a header from a heading
    pages = ["Chapter 1\nText", "Chapter 1\nOther"]
    assert remove_repeated_edge_lines(pages) == pages


def test_sample_pdf_has_no_running_header_or_page_numbers():
    doc = extract_pages(SAMPLE)
    assert "EduPulse sample notes" not in doc.text
    assert not any(line.startswith("Page ") for line in doc.text.split("\n"))
    raw = extract_pages(SAMPLE, clean=False)
    assert "EduPulse sample notes" in raw.text
