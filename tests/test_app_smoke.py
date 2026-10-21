from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app() -> AppTest:
    return AppTest.from_file(APP).run()


def test_app_renders_without_errors():
    at = run_app()
    assert not at.exception
    assert at.title[0].value == "EduPulse AI"


def test_sidebar_lists_presets():
    at = run_app()
    assert "Medical Tutor" in at.sidebar.selectbox[0].options
    assert "Socratic" in at.sidebar.selectbox[1].options


def test_add_topic_requires_text():
    at = run_app()
    at.button(key="add_topic").click().run()
    assert at.warning


def test_topics_persist_in_session_state():
    at = run_app()
    at.text_input[0].input("Biology").run()
    at.button(key="add_topic").click().run()
    assert at.session_state.topics == ["Biology"]
    assert at.metric[0].value == "1"

    # adding the same topic again should not duplicate it
    at.button(key="add_topic").click().run()
    assert at.session_state.topics == ["Biology"]
    assert at.info


def test_clear_list():
    at = run_app()
    at.text_input[0].input("Chemistry").run()
    at.button(key="add_topic").click().run()
    at.button(key="clear_topics").click().run()
    assert at.session_state.topics == []
