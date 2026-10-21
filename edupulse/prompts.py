"""Prompt templates and builders."""

from __future__ import annotations

from edupulse.presets import Persona, Tone

FORMAT_RULES = (
    "Format every answer in Markdown: short paragraphs, bullet points for lists "
    "and **bold** for key terms. Use $...$ for math. If the material is unclear "
    "or incomplete, say so instead of guessing."
)

TASKS = {
    "Explain": "Explain the following material so a student can understand it.",
    "Summarize": "Summarize the key points of the following material.",
    "Study notes": (
        "Turn the following material into structured study notes "
        "with headings and bullet points."
    ),
    "Practice questions": (
        "Write three practice questions, with answers, based on the following material."
    ),
}


def build_system_prompt(persona: Persona, tone: Tone) -> str:
    return f"{persona.instructions}\n\nTone: {tone.instructions}"


def build_user_prompt(task: str, material: str) -> str:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task!r}")
    return f"{TASKS[task]}\n\n---\n{material.strip()}\n---"


def build_chat_messages(
    system_prompt: str, history: list[dict], max_messages: int = 10
) -> list[dict]:
    """System prompt + the most recent chat turns.

    Only the last `max_messages` messages are sent so long conversations
    don't blow up the context window (and the bill). The window never
    starts with an assistant message.
    """
    recent = history[-max_messages:] if max_messages > 0 else []
    if recent and recent[0]["role"] == "assistant":
        recent = recent[1:]
    return [{"role": "system", "content": system_prompt}, *recent]
