import json

from edupulse.history import export_json, export_markdown

MESSAGES = [
    {"role": "user", "content": "What is ATP?"},
    {"role": "assistant", "content": "The cell's energy currency.", "persona": "Medical Tutor"},
]


def test_export_json_round_trip():
    data = json.loads(export_json(MESSAGES, persona="Medical Tutor", tone="Concise"))
    assert data["persona"] == "Medical Tutor"
    assert data["tone"] == "Concise"
    # extra UI-only keys are not exported
    assert data["messages"][1] == {"role": "assistant", "content": "The cell's energy currency."}


def test_export_markdown_names_speakers():
    md = export_markdown(MESSAGES, persona="Math Coach", tone="Concise")
    assert md.startswith("# EduPulse AI chat with the Math Coach")
    assert "**You:**" in md
    assert "**Medical Tutor:**" in md
