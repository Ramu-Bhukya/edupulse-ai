import pytest

from edupulse.presets import Persona, Tone
from edupulse.prompts import TASKS, build_system_prompt, build_user_prompt

PERSONA = Persona(name="Tester", description="d", instructions="You are a test tutor.")
TONE = Tone(name="Calm", instructions="Stay calm.")


def test_system_prompt_combines_persona_and_tone():
    prompt = build_system_prompt(PERSONA, TONE)
    assert "You are a test tutor." in prompt
    assert "Stay calm." in prompt


@pytest.mark.parametrize("task", list(TASKS))
def test_user_prompt_contains_task_and_material(task):
    prompt = build_user_prompt(task, "  Mitochondria make ATP.  ")
    assert prompt.startswith(TASKS[task])
    assert "Mitochondria make ATP." in prompt


def test_unknown_task_rejected():
    with pytest.raises(ValueError):
        build_user_prompt("Dance", "text")
