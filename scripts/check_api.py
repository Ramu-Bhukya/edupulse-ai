"""Check that the OpenAI key and model are working.

Usage:
    python scripts/check_api.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edupulse.config import MissingAPIKeyError, get_settings  # noqa: E402
from edupulse.llm import LLM_ERRORS, complete, friendly_error, get_client  # noqa: E402


def main() -> int:
    try:
        client = get_client()
    except MissingAPIKeyError as exc:
        print(exc)
        return 1

    model = get_settings().chat_model
    print(f"Sending a test prompt to {model}...")
    try:
        reply = complete(
            client,
            [{"role": "user", "content": "In one short sentence, what is photosynthesis?"}],
        )
    except LLM_ERRORS as exc:
        print(f"Failed: {friendly_error(exc)}")
        return 1
    print(f"Reply: {reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
