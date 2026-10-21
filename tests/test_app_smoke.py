from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from edupulse import llm
from tests.conftest import FakeOpenAI

APP = str(Path(__file__).resolve().parent.parent / "app.py")
MATERIAL = "The heart has four chambers: two atria and two ventricles."


@pytest.fixture
def fake_llm(monkeypatch):
    client = FakeOpenAI("**Answer** from the fake tutor")
    monkeypatch.setattr(llm, "get_client", lambda api_key=None: client)
    return client


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
