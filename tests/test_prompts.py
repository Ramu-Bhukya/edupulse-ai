import pytest

from edupulse.presets import Persona, Tone
from edupulse.prompts import (
    FORMAT_RULES,
    TASKS,
    build_chat_messages,
    build_system_prompt,
    build_user_prompt,
)

PERSONA = Persona(name="Tester", description="d", instructions="You are a test tutor.")
TONE = Tone(name="Calm", instructions="Stay calm.")


def test_system_prompt_combines_persona_and_tone():
    prompt = build_system_prompt(PERSONA, TONE)
    assert "You are a test tutor." in prompt
    assert "Stay calm." in prompt


def test_system_prompt_includes_formatting_rules():
    assert build_system_prompt(PERSONA, TONE).endswith(FORMAT_RULES)


@pytest.mark.parametrize("task", list(TASKS))
def test_user_prompt_contains_task_and_material(task):
    prompt = build_user_prompt(task, "  Mitochondria make ATP.  ")
    assert prompt.startswith(TASKS[task])
    assert "Mitochondria make ATP." in prompt


def test_unknown_task_rejected():
    with pytest.raises(ValueError):
        build_user_prompt("Dance", "text")


def _history(n):
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": f"m{i}"} for i in range(n)]


def test_chat_messages_start_with_system_prompt():
    messages = build_chat_messages("sys", _history(2))
    assert messages[0] == {"role": "system", "content": "sys"}
    assert [m["content"] for m in messages[1:]] == ["m0", "m1"]


def test_chat_messages_are_trimmed_to_recent_turns():
    messages = build_chat_messages("sys", _history(15), max_messages=4)
    # last 4 would start with an assistant message (m11), so it is dropped
    assert [m["content"] for m in messages[1:]] == ["m12", "m13", "m14"]
    assert messages[1]["role"] == "user"


def test_chat_messages_with_empty_history():
    assert build_chat_messages("sys", []) == [{"role": "system", "content": "sys"}]


def test_chat_messages_strip_ui_only_keys():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "persona": "Math Coach"},
    ]
    messages = build_chat_messages("sys", history)
    assert messages[2] == {"role": "assistant", "content": "hello"}
