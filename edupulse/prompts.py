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
