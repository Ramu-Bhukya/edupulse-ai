from pathlib import Path

import httpx2
import openai
import pytest
from streamlit.testing.v1 import AppTest

from edupulse import embeddings, llm
from tests.conftest import FakeOpenAI, KeywordEmbedder

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "cell_biology_notes.pdf"
MATERIAL = "The heart has four chambers: two atria and two ventricles."


@pytest.fixture
def fake_llm(monkeypatch):
    client = FakeOpenAI("**Answer** from the fake tutor")
    monkeypatch.setattr(llm, "get_client", lambda api_key=None: client)
    return client


@pytest.fixture(autouse=True)
def isolated_chroma(monkeypatch, tmp_path):
    # every test gets its own empty vector store
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))


@pytest.fixture
def fake_rag(monkeypatch, fake_llm):
    embedder = KeywordEmbedder()
    monkeypatch.setattr(embeddings, "OpenAIEmbedder", lambda client, **kwargs: embedder)
    return embedder


def upload_sample(at: AppTest) -> AppTest:
    return at.sidebar.file_uploader[0].upload(
        "cells.pdf", SAMPLE.read_bytes(), "application/pdf"
    ).run()


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=15).run()


def generate(at: AppTest, material: str = MATERIAL) -> AppTest:
    at.text_area[0].input(material).run()
    return at.button(key="generate").click().run()


def test_app_renders_without_errors():
    at = run_app()
    assert not at.exception
    assert at.title[0].value == "EduPulse AI"


def test_sidebar_lists_presets():
    at = run_app()
    assert "Medical Tutor" in at.sidebar.selectbox[0].options
    assert "Socratic" in at.sidebar.selectbox[1].options


def test_generate_requires_material(fake_llm):
    at = run_app()
    at.button(key="generate").click().run()
    assert at.warning
    assert at.session_state.generated is None
    assert fake_llm.chat.completions.calls == []


def test_generate_rejects_very_short_material(fake_llm):
    at = generate(run_app(), "hi")
    assert at.warning
    assert at.session_state.history == []


def test_missing_key_shows_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = generate(run_app())
    assert not at.exception
    assert "API key" in at.error[0].value


def test_generate_calls_llm_with_persona(fake_llm):
    at = run_app()
    at.sidebar.selectbox[0].select("Medical Tutor").run()
    at = generate(at)
    result = at.session_state.generated
    assert result["persona"] == "Medical Tutor"
    assert result["response"] == "**Answer** from the fake tutor"

    sent = fake_llm.chat.completions.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user"]
    assert "medical tutor" in sent[0]["content"].lower()
    assert MATERIAL in sent[1]["content"]


def test_history_keeps_newest_first(fake_llm):
    at = generate(run_app())
    at.main.selectbox[0].select("Summarize").run()
    at.button(key="generate").click().run()
    history = at.session_state.history
    assert [h["task"] for h in history] == ["Summarize", "Explain"]


def test_chat_streams_reply(fake_llm):
    at = run_app()
    at.chat_input[0].set_value("What is a ventricle?").run()
    assert not at.exception
    assert len(at.chat_message) == 2
    assert at.chat_message[0].markdown[0].value == "What is a ventricle?"
    assert "fake tutor" in at.chat_message[1].markdown[0].value
    sent = fake_llm.chat.completions.calls[0]
    assert sent["stream"] is True
    assert sent["messages"][-1] == {"role": "user", "content": "What is a ventricle?"}


def test_chat_remembers_previous_turns(fake_llm):
    at = run_app()
    at.chat_input[0].set_value("What is a ventricle?").run()
    at.chat_input[0].set_value("And an atrium?").run()

    assert [m["role"] for m in at.session_state.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert len(at.chat_message) == 4
    second_call = fake_llm.chat.completions.calls[1]["messages"]
    assert [m["role"] for m in second_call] == ["system", "user", "assistant", "user"]


def test_clear_chat(fake_llm):
    at = run_app()
    at.chat_input[0].set_value("What is a ventricle?").run()
    at.button(key="clear_chat").click().run()
    assert at.session_state.messages == []
    assert len(at.chat_message) == 0


def test_chat_without_key_does_not_store_question(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = run_app()
    at.chat_input[0].set_value("Hello?").run()
    assert at.session_state.messages == []
    assert "API key" in at.error[0].value


def test_key_takeaways_needs_conversation(fake_llm):
    at = run_app()
    at.button(key="get_takeaways").click().run()
    assert any("Ask a few questions first" in info.value for info in at.info)
    assert at.session_state.takeaways is None


def test_key_takeaways_rendered_from_structured_output(fake_llm):
    fake_llm.chat.completions.parsed = {"title": "Heart", "points": ["Four chambers."]}
    at = run_app()
    at.chat_input[0].set_value("What is a ventricle?").run()
    at.button(key="get_takeaways").click().run()
    assert not at.exception
    assert at.session_state.takeaways.points == ["Four chambers."]
    assert any("Four chambers." in md.value for md in at.markdown)


def test_mentor_tracks_persona_and_exports(fake_llm):
    at = run_app()
    at.sidebar.selectbox[0].select("Math Coach").run()
    at.chat_input[0].set_value("What is a derivative?").run()
    assert at.session_state.messages[1]["persona"] == "Math Coach"
    assert at.sidebar.metric[0].value == "2"
    labels = [b.label for b in at.sidebar.get("download_button")]
    assert labels == ["Export chat (.json)", "Export chat (.md)"]


def test_welcome_message_only_when_empty(fake_llm):
    at = run_app()
    assert any("Hi! I'm your" in info.value for info in at.info)
    at.chat_input[0].set_value("Hello").run()
    assert not any("Hi! I'm your" in info.value for info in at.info)


def test_api_errors_are_shown_not_raised(monkeypatch):
    class BrokenCompletions:
        def create(self, **kwargs):
            request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
            response = httpx2.Response(401, request=request)
            raise openai.AuthenticationError("bad key", response=response, body=None)

    client = FakeOpenAI()
    client.chat.completions = BrokenCompletions()
    monkeypatch.setattr(llm, "get_client", lambda api_key=None: client)

    at = run_app()
    at.chat_input[0].set_value("Hello?").run()
    assert not at.exception
    assert "rejected the API key" in at.error[0].value
    assert at.session_state.messages == []


def test_sidebar_key_is_passed_to_client(monkeypatch):
    seen = []
    client = FakeOpenAI()

    def fake_get_client(api_key=None):
        seen.append(api_key)
        return client

    monkeypatch.setattr(llm, "get_client", fake_get_client)
    at = run_app()
    at.sidebar.text_input[0].input("sk-user-supplied-key").run()
    at.chat_input[0].set_value("Hello?").run()
    assert seen[-1] == "sk-user-supplied-key"
    assert "sk-...-key" in at.sidebar.caption[-1].value


def test_knowledge_base_needs_a_document():
    at = run_app()
    assert any("Upload a PDF in the sidebar" in info.value for info in at.info)


def test_upload_indexes_pdf_once(fake_rag):
    at = upload_sample(run_app())
    assert not at.exception
    doc = at.session_state.document
    assert doc.filename == "cells.pdf"
    assert doc.page_count == 3
    assert "3 pages" in at.sidebar.success[0].value

    # further reruns must not index again
    at.run()
    assert fake_rag.calls == 1


def test_knowledge_base_search(fake_rag):
    at = upload_sample(run_app())
    at.text_input(key="kb_query").input("What does the Golgi apparatus do?")
    at.button(key="FormSubmitter:kb_search-Search").click().run()
    assert not at.exception
    results = at.session_state.search_results
    assert len(results) == 3
    assert "Golgi" in results[0].chunk.text


def test_invalid_pdf_shows_error(fake_rag):
    at = run_app()
    at.sidebar.file_uploader[0].upload("broken.pdf", b"not a pdf", "application/pdf").run()
    assert not at.exception
    assert "valid PDF" in at.sidebar.error[0].value
    assert at.session_state.document is None


def test_reupload_of_known_file_uses_existing_index(fake_rag):
    first = upload_sample(run_app())
    assert fake_rag.calls == 1
    # a brand new session (same Chroma dir) uploads the same file again
    second = upload_sample(run_app())
    assert fake_rag.calls == 1
    assert second.session_state.document == first.session_state.document
    assert "indexed before" in second.toast[0].value


def test_reopen_document_from_library(fake_rag):
    upload_sample(run_app())
    at = run_app()  # new session, nothing uploaded
    picker = at.sidebar.selectbox(key="library_choice")
    picker.select(picker.options[1]).run()
    assert at.session_state.document.filename == "cells.pdf"


def test_remove_document_from_library(fake_rag):
    doc_id = upload_sample(run_app()).session_state.document.doc_id
    at = run_app()
    at.sidebar.selectbox(key="library_choice").set_value(doc_id).run()
    at.button(key="remove_doc").click().run()
    assert at.session_state.document is None
    # the library is empty now, so the picker is gone
    assert not [box for box in at.sidebar.selectbox if box.key == "library_choice"]


def test_search_results_show_score_and_highlights(fake_rag):
    at = upload_sample(run_app())
    at.text_input(key="kb_query").input("Golgi apparatus")
    at.button(key="FormSubmitter:kb_search-Search").click().run()
    assert at.get("progress")
    assert any("**Golgi**" in md.value for md in at.markdown)


def test_chat_uses_document_context_when_loaded(fake_rag, fake_llm):
    at = upload_sample(run_app())
    at.chat_input[0].set_value("What does the Golgi apparatus do?").run()
    assert not at.exception
    system = fake_llm.chat.completions.calls[-1]["messages"][0]["content"]
    assert "<context>" in system
    assert "ONLY the context" in system
    assert "Golgi apparatus" in system

    answer = at.session_state.messages[-1]
    assert answer["document"] == "cells.pdf"
    assert answer["sources"] and "page_label" in answer["sources"][0]
    assert any(e.label.startswith("Sources (") for e in at.expander)


def test_chat_without_document_has_no_context(fake_llm):
    at = run_app()
    at.chat_input[0].set_value("What is a cell?").run()
    system = fake_llm.chat.completions.calls[-1]["messages"][0]["content"]
    assert "<context>" not in system
    assert "sources" not in at.session_state.messages[-1]


def test_idk_answer_shows_hint_instead_of_sources(fake_rag, fake_llm):
    fake_llm.chat.completions.reply = "I don't know."
    at = upload_sample(run_app())
    at.chat_input[0].set_value("Who won the 1998 World Cup?").run()
    assert any("doesn't seem to be covered" in c.value for c in at.caption)
    assert not [e for e in at.expander if e.label.startswith("Sources")]


def test_main_tabs():
    at = run_app()
    labels = [tab.label for tab in at.tabs]
    assert labels[:3] == ["💬 Chat with Notes", "📝 Quiz Generator", "🃏 Flashcards"]
    assert any("quiz you on it" in info.value for info in at.info)
