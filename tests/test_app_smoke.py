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
    return AppTest.from_file(APP).run()


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
