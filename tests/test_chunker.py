from pathlib import Path

import pytest

from edupulse.chunker import Chunk, chunk_document, split_text
from edupulse.pdf_parser import Page, ParsedDocument, extract_pages

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "cell_biology_notes.pdf"

TEXT = " ".join(
    f"Sentence number {i} talks about cells, membranes and transport." for i in range(60)
)


def chunks_of(text, size, overlap):
    return [text[s:e] for s, e in split_text(text, size, overlap)]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("abc", size, overlap)


def test_empty_and_short_text():
    assert split_text("", 100, 10) == []
    assert split_text("   \n  ", 100, 10) == []
    assert chunks_of("Just one short sentence.", 100, 10) == ["Just one short sentence."]


def test_chunks_respect_max_size():
    for chunk in chunks_of(TEXT, 300, 50):
        assert 0 < len(chunk) <= 300


def test_chunks_end_on_sentence_boundaries():
    chunks = chunks_of(TEXT, 300, 50)
    for chunk in chunks:
        assert chunk.endswith(".")


def test_chunks_start_on_word_boundaries():
    for start, _ in split_text(TEXT, 300, 50):
        assert start == 0 or TEXT[start - 1].isspace()


def test_consecutive_chunks_overlap():
    spans = split_text(TEXT, 300, 80)
    for (s1, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert s2 < e1  # the next chunk starts before the previous one ended
        assert s2 > s1  # ...but the window always moves forward


def test_no_overlap_still_covers_everything():
    chunks = chunks_of(TEXT, 250, 0)
    words = set(TEXT.split())
    covered = set(" ".join(chunks).split())
    assert words == covered


def test_hard_cut_when_there_is_no_whitespace():
    blob = "x" * 1000
    chunks = chunks_of(blob, 300, 50)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks).count("x") >= 1000


def test_chunk_document_tracks_pages():
    doc = ParsedDocument(
        filename="t.pdf",
        pages=[
            Page(1, "Alpha " * 60),
            Page(2, ""),  # blank page is skipped
            Page(3, "Beta " * 60),
        ],
    )
    chunks = chunk_document(doc, chunk_size=200, overlap=40)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].page == 1
    assert chunks[-1].page_end == 3
    assert all(c.page in (1, 3) for c in chunks)
    spanning = [c for c in chunks if c.page != c.page_end]
    assert all(c.page_label == "pp. 1-3" for c in spanning)


def test_page_label():
    assert Chunk("t", 0, 2, 2).page_label == "p. 2"
    assert Chunk("t", 0, 2, 3).page_label == "pp. 2-3"


def test_sample_pdf_chunks():
    chunks = chunk_document(extract_pages(SAMPLE))
    assert len(chunks) > 3
    assert chunks[0].page == 1
    assert max(c.page_end for c in chunks) == 3
    assert all(len(c.text) <= 1000 for c in chunks)


def test_numbered_headings_are_not_sentence_ends():
    text = ("Intro sentence here. " * 10) + "5. Cell Division " + ("More words follow. " * 10)
    for chunk in chunks_of(text, 230, 40):
        assert not chunk.endswith("5.")


def test_chunks_after_the_first_start_at_a_sentence():
    for chunk in chunks_of(TEXT, 300, 80)[1:]:
        assert chunk.startswith("Sentence number")
