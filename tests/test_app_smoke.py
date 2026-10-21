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


def test_generate_requires_material():
    at = run_app()
    at.button(key="generate").click().run()
    assert at.warning
    assert at.session_state.generated is None


def test_generate_builds_prompt():
    at = run_app()
    at.sidebar.selectbox[0].select("Medical Tutor").run()
    at.text_area[0].input("The heart has four chambers.").run()
    at.button(key="generate").click().run()
    result = at.session_state.generated
    assert result["persona"] == "Medical Tutor"
    assert "The heart has four chambers." in result["user"]
    assert len(at.code) == 2
