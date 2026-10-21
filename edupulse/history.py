"""Export chat history to JSON or Markdown."""

from __future__ import annotations

import json
from datetime import datetime


def export_json(messages: list[dict], *, persona: str, tone: str) -> str:
    payload = {
        "app": "EduPulse AI",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "persona": persona,
        "tone": tone,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def export_markdown(messages: list[dict], *, persona: str, tone: str) -> str:
    lines = [f"# EduPulse AI chat with the {persona} ({tone})", ""]
    for m in messages:
        speaker = "You" if m["role"] == "user" else m.get("persona", persona)
        lines += [f"**{speaker}:**", "", m["content"], ""]
    return "\n".join(lines)
