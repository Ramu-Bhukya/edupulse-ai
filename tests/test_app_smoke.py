from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders_without_errors():
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert at.title[0].value == "EduPulse AI"


def test_start_button_needs_subject():
    at = AppTest.from_file(APP).run()
    at.button[0].click().run()
    assert at.warning


def test_start_button_greets_subject():
    at = AppTest.from_file(APP).run()
    at.text_input[0].input("Biology").run()
    at.button[0].click().run()
    assert "Biology" in at.success[0].value
