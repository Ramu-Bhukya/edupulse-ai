from streamlit.testing.v1 import AppTest

from edupulse.chunker import Chunk
from edupulse.history import prompt_result_as_text
from edupulse.rag_engine import source_to_dict
from edupulse.study_tools import QuizQuestion
from edupulse.ui_components import DEFAULT_STATE, graded_options, snippet
from edupulse.vector_store import SearchResult


def test_graded_options_mark_answer_and_pick():
    q = QuizQuestion(
        question="Q?", options=["A", "B", "C", "D"], correct_index=1, explanation="e"
    )
    assert graded_options(q, chosen=2) == [
        "⚪ A",
        "✅ **B**",
        "❌ C (your answer)",
        "⚪ D",
    ]
    assert graded_options(q, chosen=1)[1] == "✅ **B** (your answer)"


def test_snippet_flattens_and_truncates():
    assert snippet("a\n b   c") == "a b c"
    long = "word " * 100
    assert snippet(long, limit=20) == "word word word word..."


def test_source_to_dict():
    result = SearchResult(Chunk("Text", 4, 2, 3), distance=0.25)
    assert source_to_dict(result) == {
        "index": 4,
        "page_label": "pp. 2-3",
        "text": "Text",
        "similarity": 0.75,
    }


def test_prompt_result_as_text():
    text = prompt_result_as_text({"system": "S", "user": "U", "response": "R"})
    assert text == "# System\nS\n\n# User\nU\n\n# Response\nR\n"


def test_init_session_state_gives_each_session_its_own_lists():
    def script():
        from edupulse.ui_components import init_session_state

        init_session_state()

    first = AppTest.from_function(script).run()
    second = AppTest.from_function(script).run()
    first.session_state.messages.append({"role": "user", "content": "hi"})
    assert second.session_state.messages == []
    assert all(key in first.session_state for key in DEFAULT_STATE)


def test_chat_message_with_sources_and_idk():
    def script():
        from edupulse.ui_components import render_chat_message

        render_chat_message(
            {
                "role": "assistant",
                "content": "ATP [1].",
                "persona": "Medical Tutor",
                "document": "cells.pdf",
                "sources": [
                    {"page_label": "p. 1", "text": "Mitochondria $ATP$", "similarity": 0.8}
                ],
            }
        )
        render_chat_message(
            {"role": "assistant", "content": "I don't know.", "persona": "X", "document": "a.pdf"}
        )

    at = AppTest.from_function(script).run()
    assert not at.exception
    assert at.caption[0].value == "Medical Tutor · from cells.pdf"
    assert at.expander[0].label == "Sources (1)"
    assert "\\$ATP\\$" in at.expander[0].caption[0].value  # escaped, not LaTeX
    assert any("Search tab" in c.value for c in at.caption)
