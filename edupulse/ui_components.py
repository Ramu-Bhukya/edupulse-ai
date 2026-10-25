"""Reusable Streamlit widgets for app.py.

Only presentation lives here; the app decides what to show and when.
"""

from __future__ import annotations

import html

import streamlit as st

from edupulse.rag_engine import is_idk
from edupulse.study_tools import Flashcard, QuizQuestion
from edupulse.text_utils import escape_markdown, highlight_terms
from edupulse.vector_store import DocumentInfo, SearchResult

AVATARS = {"user": "🧑‍🎓", "assistant": "🎓"}
SNIPPET_CHARS = 300

# Everything the app keeps in st.session_state, with its starting value.
DEFAULT_STATE = {
    # Prompt Architect
    "generated": None,
    "history": [],
    # chat
    "messages": [],
    "takeaways": None,
    # active document + search
    "document": None,
    "search_results": None,
    "search_query": "",
    "upload_attempt": None,
    "upload_key": 0,  # bumped to give the file uploader a fresh, empty widget
    # quiz
    "quiz": None,
    "quiz_answers": [],
    "quiz_submitted": False,
    "quiz_round": 0,  # bumped for every new quiz so the radio keys reset
    # flashcards
    "deck": None,
    "card_order": [],
    "card_pos": 0,
    "card_flipped": False,
}

CARD_CSS = """
<style>
.flashcard {
    min-height: 220px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
    margin-bottom: 1rem;
}
.flashcard .side { font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; }
.flashcard .content { font-size: 1.5rem; margin-top: 0.75rem; }
.flashcard.front { background: #eef2ff; color: #1e1b4b; }
.flashcard.back { background: #ecfdf5; color: #064e3b; }
.flashcard.back .content { font-size: 1.15rem; }
</style>
"""


def init_session_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            # copy lists so sessions never share the same object
            st.session_state[key] = list(value) if isinstance(value, list) else value


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "..."


# --- chat -------------------------------------------------------------------------


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for i, source in enumerate(sources, start=1):
            st.markdown(f"**[{i}]** {source['page_label']} · {source['similarity']:.0%} match")
            st.caption(escape_markdown(snippet(source["text"])))


def render_chat_message(message: dict) -> None:
    role = message["role"]
    with st.chat_message(role, avatar=AVATARS[role]):
        if role == "assistant":
            label = message.get("persona", "")
            if message.get("document"):
                label += f" · from {message['document']}"
            st.caption(label)
        st.markdown(message["content"])
        if message.get("document") and is_idk(message["content"]):
            st.caption(
                "That doesn't seem to be covered in your document (with a PDF open, "
                "answers come only from it). Try rephrasing or the Search tab, or click "
                "**Chat without a document** in the sidebar to ask general questions."
            )
        elif message.get("sources"):
            render_sources(message["sources"])


# --- documents and search -----------------------------------------------------------


def render_document_badge(doc: DocumentInfo) -> None:
    st.success(f"**{doc.filename}**  \n{doc.page_count} pages · {doc.chunk_count} chunks")


def render_search_result(rank: int, result: SearchResult, query: str) -> None:
    with st.container(border=True):
        title_col, score_col = st.columns([3, 2])
        title_col.markdown(
            f"**#{rank}** · {result.chunk.page_label} · chunk {result.chunk.index + 1}"
        )
        score = max(0.0, min(1.0, result.similarity))
        score_col.progress(score, text=f"{score:.0%} match")
        st.markdown(highlight_terms(result.chunk.text, query))


# --- quiz and flashcards -------------------------------------------------------------


def render_quiz_feedback(question: QuizQuestion, chosen: int | None) -> None:
    if chosen == question.correct_index:
        st.success(f"Correct! {question.explanation}")
    else:
        st.error(f"Not quite. The answer is **{question.correct_option}**. {question.explanation}")


def graded_options(question: QuizQuestion, chosen: int | None) -> list[str]:
    """One Markdown line per option, marking the right answer and the student's pick."""
    lines = []
    for i, option in enumerate(question.options):
        text = escape_markdown(option)
        if i == question.correct_index:
            line = f"✅ **{text}**"
        elif i == chosen:
            line = f"❌ {text}"
        else:
            line = f"⚪ {text}"
        if i == chosen:
            line += " (your answer)"
        lines.append(line)
    return lines


def render_graded_question(number: int, question: QuizQuestion, chosen: int | None) -> None:
    # Read-only review: disabled radios lost the selection after the rerun,
    # so students couldn't see what they had picked.
    st.markdown(f"**{number}. {escape_markdown(question.question)}**")
    st.markdown("  \n".join(graded_options(question, chosen)))
    render_quiz_feedback(question, chosen)


def render_flashcard(card: Flashcard, flipped: bool) -> None:
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    side, text = ("Definition", card.definition) if flipped else ("Term", card.term)
    st.markdown(
        f'<div class="flashcard {"back" if flipped else "front"}">'
        f'<div class="side">{side}</div>'
        f'<div class="content">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )
