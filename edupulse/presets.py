"""Persona and tone presets used to build system prompts.

The presets live in data/personas.json so new personas can be added
without touching the code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRESETS_FILE = DATA_DIR / "personas.json"


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    instructions: str


@dataclass(frozen=True)
class Tone:
    name: str
    instructions: str


def load_presets(path: Path = PRESETS_FILE) -> tuple[list[Persona], list[Tone]]:
    """Read personas and tones from a JSON file."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    try:
        personas = [Persona(**item) for item in raw["personas"]]
        tones = [Tone(**item) for item in raw["tones"]]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid presets file {path}: {exc}") from exc

    if not personas or not tones:
        raise ValueError(f"Presets file {path} needs at least one persona and one tone")

    return personas, tones


def find_by_name(items: list, name: str):
    """Return the preset whose name matches (case-insensitive), or None."""
    name = name.strip().lower()
    return next((item for item in items if item.name.lower() == name), None)
