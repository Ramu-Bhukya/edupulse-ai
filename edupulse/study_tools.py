"""Study helpers that need structured (JSON) answers from the model."""

from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel

from edupulse.llm import parse_structured


class KeyTakeaways(BaseModel):
    title: str
    points: list[str]


TAKEAWAYS_PROMPT = (
    "You summarize tutoring conversations for a student. Return a short title "
    "and 3 to 6 key takeaways. Each takeaway is one sentence, written for revision."
)


def summarize_takeaways(client: OpenAI, history: list[dict]) -> KeyTakeaways:
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    messages = [
        {"role": "system", "content": TAKEAWAYS_PROMPT},
        {"role": "user", "content": f"Conversation:\n{transcript}"},
    ]
    takeaways = parse_structured(client, messages, KeyTakeaways)
    takeaways.points = [p.strip() for p in takeaways.points if p.strip()]
    return takeaways
