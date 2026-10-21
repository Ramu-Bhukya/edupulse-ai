import random

import pytest

from edupulse.chunker import Chunk
from edupulse.llm import StructuredOutputError
from edupulse.study_tools import (
    KeyTakeaways,
    Quiz,
    QuizQuestion,
    evenly_spaced,
    format_chunks,
    generate_quiz,
    sample_context,
    shuffle_options,
    summarize_takeaways,
    validate_quiz,
)
from edupulse.vector_store import VectorStore


def test_summarize_takeaways(fake_client):
    fake_client.chat.completions.parsed = {
        "title": "Heart basics",
        "points": ["The heart has four chambers.", "  ", " Atria receive blood. "],
    }
    history = [
        {"role": "user", "content": "How many chambers?"},
        {"role": "assistant", "content": "Four."},
    ]
    result = summarize_takeaways(fake_client, history)
    assert isinstance(result, KeyTakeaways)
    assert result.points == ["The heart has four chambers.", "Atria receive blood."]

    sent = fake_client.chat.completions.calls[0]["messages"]
    assert "USER: How many chambers?" in sent[1]["content"]
    assert "ASSISTANT: Four." in sent[1]["content"]


# --- quiz ----------------------------------------------------------------------


def question(text="What makes ATP?", options=None, correct=0, explanation="Because [p. 1]."):
    return {
        "question": text,
        "options": options or ["Mitochondria", "Ribosomes", "Golgi", "Nucleus"],
        "correct_index": correct,
        "explanation": explanation,
    }


def test_evenly_spaced():
    items = list(range(10))
    assert evenly_spaced(items, 3) == [0, 4, 9]  # round(4.5) == 4
    assert evenly_spaced(items, 20) == items
    assert evenly_spaced(items, 1) == [5]
    assert evenly_spaced([], 3) == []
    assert evenly_spaced(items, 0) == []


def test_shuffle_keeps_the_right_answer():
    q = QuizQuestion(**question(correct=2))
    for seed in range(10):
        shuffled = shuffle_options(q, random.Random(seed))
        assert shuffled.correct_option == "Golgi"
        assert sorted(shuffled.options) == sorted(q.options)


def test_validate_drops_malformed_questions():
    quiz = Quiz(
        questions=[
            QuizQuestion(**question()),
            QuizQuestion(**question(options=["a", "b", "c"])),  # only 3 options
            QuizQuestion(**question(correct=4)),  # index out of range
            QuizQuestion(**question(options=["a", "a", "b", "c"])),  # duplicate option
            QuizQuestion(**question(text="   ")),  # empty question
            QuizQuestion(**question(options=["a", " ", "b", "c"])),  # blank option
        ]
    )
    result = validate_quiz(quiz, n=5, rng=random.Random(0))
    assert len(result.questions) == 1
    assert result.questions[0].correct_option == "Mitochondria"


def test_validate_limits_to_n_and_strips_whitespace():
    quiz = Quiz(questions=[QuizQuestion(**question(text=f"  Q{i}?  ")) for i in range(7)])
    result = validate_quiz(quiz, n=5)
    assert [q.question for q in result.questions] == ["Q0?", "Q1?", "Q2?", "Q3?", "Q4?"]


def test_validate_with_nothing_usable():
    with pytest.raises(StructuredOutputError):
        validate_quiz(Quiz(questions=[QuizQuestion(**question(correct=9))]), n=5)


def test_generate_quiz_sends_material(fake_client):
    fake_client.chat.completions.parsed = {"questions": [question() for _ in range(5)]}
    chunks = [Chunk("Mitochondria produce ATP.", 0, 3, 3)]
    quiz = generate_quiz(fake_client, chunks, n=5, rng=random.Random(1))
    assert len(quiz.questions) == 5

    call = fake_client.chat.completions.calls[0]
    assert call["response_format"] is Quiz
    assert "exactly 5 questions" in call["messages"][0]["content"]
    assert "[p. 3]\nMitochondria produce ATP." in call["messages"][1]["content"]


def test_generate_quiz_needs_material(fake_client):
    with pytest.raises(ValueError):
        generate_quiz(fake_client, [])


def test_format_chunks():
    text = format_chunks([Chunk("A", 0, 1, 1), Chunk("B", 1, 1, 2)])
    assert text == "[p. 1]\nA\n\n[pp. 1-2]\nB"


@pytest.fixture
def store(tmp_path, keyword_embedder):
    store = VectorStore.persistent(str(tmp_path / "chroma"))
    chunks = [Chunk(f"Chunk {i} about topic{i}.", i, i + 1, i + 1) for i in range(20)]
    chunks[13] = Chunk("Osmosis moves water across membranes.", 13, 14, 14)
    store.add_document("d", "d.pdf", chunks, keyword_embedder.embed([c.text for c in chunks]), 20)
    return store


def test_sample_context_spreads_over_document(store, keyword_embedder):
    chunks = sample_context(store, keyword_embedder, "d", max_chunks=4)
    assert [c.index for c in chunks] == [0, 6, 13, 19]


def test_sample_context_with_topic(store, keyword_embedder):
    chunks = sample_context(store, keyword_embedder, "d", topic="osmosis water")
    assert 13 in [c.index for c in chunks]
    assert [c.index for c in chunks] == sorted(c.index for c in chunks)


def test_sample_context_respects_char_budget(store, keyword_embedder):
    # chunk texts are ~21 chars, so only two fit in 50
    chunks = sample_context(store, keyword_embedder, "d", max_chunks=10, max_chars=50)
    assert len(chunks) == 2
    assert sum(len(c.text) for c in chunks) <= 50


def test_sample_context_always_returns_at_least_one_chunk(store, keyword_embedder):
    assert len(sample_context(store, keyword_embedder, "d", max_chars=1)) == 1
