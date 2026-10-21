"""App configuration and API key handling.

Secrets are never hard-coded. The OpenAI key is looked up in this order:
1. a key passed in explicitly (e.g. typed into the sidebar)
2. the OPENAI_API_KEY environment variable (loaded from .env locally)
3. Streamlit secrets (used on Streamlit Community Cloud)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHROMA_DIR = ".chroma"


class MissingAPIKeyError(RuntimeError):
    """Raised when no OpenAI API key can be found."""


@dataclass(frozen=True)
class Settings:
    chat_model: str
    embed_model: str
    temperature: float
    chroma_dir: str


def get_settings() -> Settings:
    try:
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    except ValueError:
        temperature = 0.3
    return Settings(
        chat_model=os.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL),
        embed_model=os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL),
        temperature=temperature,
        chroma_dir=os.getenv("CHROMA_DIR", DEFAULT_CHROMA_DIR),
    )


def _streamlit_secret(name: str) -> str | None:
    # Only look at st.secrets when we are actually running inside Streamlit,
    # so importing this module from scripts/tests doesn't pull Streamlit in.
    st = sys.modules.get("streamlit")
    if st is None:
        return None
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets.toml -> Streamlit raises
        return None


def get_api_key(override: str | None = None) -> str:
    """Return the OpenAI API key or raise MissingAPIKeyError."""
    candidates = (override, os.getenv("OPENAI_API_KEY"), _streamlit_secret("OPENAI_API_KEY"))
    for key in candidates:
        if key and key.strip():
            return key.strip()
    raise MissingAPIKeyError(
        "No OpenAI API key found. Add OPENAI_API_KEY to your .env file "
        "(see .env.example) or enter a key in the sidebar."
    )


def mask_key(key: str) -> str:
    """Show only the start and end of a key, e.g. for status messages."""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:3]}...{key[-4:]}"
