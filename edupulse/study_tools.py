"""Study helpers that need structured (JSON) answers from the model."""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel

from edupulse.chunker import Chunk
from edupulse.embeddings import Embedder
from edupulse.llm import StructuredOutputError, parse_structured
from edupulse.vector_store import VectorStore

MAX_CONTEXT_CHUNKS = 8
MAX_CONTEXT_CHARS = 12_000
TOPIC_TOP_K = 6


# --- key takeaways ---------------------------------------------------------------


class KeyTakeaways(BaseModel):
    title: str
    points: list[str]


TAKEAWAYS_PROMPT = (
    "You summarize tutoring conversations for a student. Return a short title "
    "and 3 to 6 key takeaways. Each takeaway is one sentence, written for revision."
)


def summarize_takeaways(client: OpenAI, history: list[dict]) -> KeyTakeaways:
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    messages = [
        {"role": "system", "content": TAKEAWAYS_PROMPT},
        {"role": "user", "content": f"Conversation:\n{transcript}"},
    ]
    takeaways = parse_structured(client, messages, KeyTakeaways)
    takeaways.points = [p.strip() for p in takeaways.points if p.strip()]
    return takeaways


# --- picking study material from the document --------------------------------------


def evenly_spaced(items: list, n: int) -> list:
    """Pick up to n items spread across the whole list (keeps first and last)."""
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def sample_context(
    store: VectorStore,
    embedder: Embedder,
    doc_id: str,
    topic: str | None = None,
    *,
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> list[Chunk]:
    """Chunks to build questions from.

    With a topic, use the chunks most relevant to it. Without one, take
    chunks spread across the whole document so questions don't all come
    from chapter 1.
    """
    if topic and topic.strip():
        results = store.search(doc_id, embedder.embed_query(topic), k=TOPIC_TOP_K)
        chunks = sorted((r.chunk for r in results), key=lambda c: c.index)
    else:
        chunks = evenly_spaced(store.get_chunks(doc_id), max_chunks)

    picked, total = [], 0
    for chunk in chunks:
        if picked and total + len(chunk.text) > max_chars:
            break
        picked.append(chunk)
        total += len(chunk.text)
    return picked


def format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.page_label}]\n{c.text}" for c in chunks)


# --- quiz ------------------------------------------------------------------------


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_index: int
    explanation: str

    @property
    def correct_option(self) -> str:
        return self.options[self.correct_index]


class Quiz(BaseModel):
    questions: list[QuizQuestion]


QUIZ_PROMPT = (
    "You write multiple-choice questions that help a student revise. Use ONLY the study "
    "material provided, never outside facts. Write exactly {n} questions covering different "
    "parts of the material. Each question has exactly 4 short, plausible options and one "
    "correct answer (correct_index is 0-3). Avoid 'all of the above' and 'none of the above'. "
    "The explanation says in one or two sentences why the answer is right, based on the "
    "material."
)


def _valid_question(q: QuizQuestion) -> bool:
    options = [o.strip() for o in q.options]
    return (
        bool(q.question.strip())
        and len(options) == 4
        and all(options)
        and len({o.lower() for o in options}) == 4
        and 0 <= q.correct_index < 4
    )


def shuffle_options(question: QuizQuestion, rng: random.Random) -> QuizQuestion:
    """Shuffle the options (models like putting the answer first) and fix the index."""
    order = list(range(len(question.options)))
    rng.shuffle(order)
    return question.model_copy(
        update={
            "options": [question.options[i] for i in order],
            "correct_index": order.index(question.correct_index),
        }
    )


def validate_quiz(quiz: Quiz, n: int, rng: random.Random | None = None) -> Quiz:
    """Drop malformed questions, tidy whitespace, shuffle options, keep at most n."""
    rng = rng or random.Random()
    clean = []
    for q in quiz.questions:
        q = q.model_copy(
            update={
                "question": q.question.strip(),
                "options": [o.strip() for o in q.options],
                "explanation": q.explanation.strip(),
            }
        )
        if _valid_question(q):
            clean.append(shuffle_options(q, rng))
    if not clean:
        raise StructuredOutputError("The model didn't return any usable quiz questions.")
    return Quiz(questions=clean[:n])


def generate_quiz(
    client: OpenAI, chunks: list[Chunk], n: int = 5, rng: random.Random | None = None
) -> Quiz:
    if not chunks:
        raise ValueError("No study material to build a quiz from.")
    messages = [
        {"role": "system", "content": QUIZ_PROMPT.format(n=n)},
        {"role": "user", "content": f"Study material:\n\n{format_chunks(chunks)}"},
    ]
    return validate_quiz(parse_structured(client, messages, Quiz), n, rng)


@dataclass(frozen=True)
class QuizResult:
    correct: list[bool]

    @property
    def score(self) -> int:
        return sum(self.correct)

    @property
    def total(self) -> int:
        return len(self.correct)

    @property
    def percent(self) -> float:
        return self.score / self.total if self.total else 0.0


def grade_quiz(quiz: Quiz, answers: list[int | None]) -> QuizResult:
    """Compare the chosen option indexes with the answer key (None = unanswered)."""
    if len(answers) != len(quiz.questions):
        raise ValueError(f"Expected {len(quiz.questions)} answers, got {len(answers)}")
    return QuizResult(
        correct=[a == q.correct_index for q, a in zip(quiz.questions, answers, strict=True)]
    )


# --- flashcards ------------------------------------------------------------------


class Flashcard(BaseModel):
    term: str
    definition: str


class FlashcardDeck(BaseModel):
    cards: list[Flashcard]


FLASHCARD_PROMPT = (
    "You make revision flashcards from the study material. Use ONLY the material provided. "
    "Write exactly {n} cards about its most important terms and concepts. 'term' is the key "
    "term or a short question (at most 8 words). 'definition' explains it in one or two "
    "plain sentences. Don't repeat a term."
)


def validate_deck(deck: FlashcardDeck, n: int) -> FlashcardDeck:
    """Strip whitespace, drop empty or duplicate cards, keep at most n."""
    seen, cards = set(), []
    for card in deck.cards:
        term, definition = card.term.strip(), card.definition.strip()
        if not term or not definition or term.lower() in seen:
            continue
        seen.add(term.lower())
        cards.append(Flashcard(term=term, definition=definition))
    if not cards:
        raise StructuredOutputError("The model didn't return any usable flashcards.")
    return FlashcardDeck(cards=cards[:n])


def generate_flashcards(client: OpenAI, chunks: list[Chunk], n: int = 10) -> FlashcardDeck:
    if not chunks:
        raise ValueError("No study material to build flashcards from.")
    messages = [
        {"role": "system", "content": FLASHCARD_PROMPT.format(n=n)},
        {"role": "user", "content": f"Study material:\n\n{format_chunks(chunks)}"},
    ]
    return validate_deck(parse_structured(client, messages, FlashcardDeck), n)


def deck_to_csv(deck: FlashcardDeck) -> str:
    """Two-column CSV (term, definition) that Anki and Quizlet can import."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for card in deck.cards:
        writer.writerow([card.term, card.definition])
    return buffer.getvalue()
