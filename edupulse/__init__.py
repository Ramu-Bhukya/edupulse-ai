"""EduPulse AI core package."""

import sqlite3
import sys

__version__ = "1.0.0"

# ChromaDB needs SQLite >= 3.35. Some Linux hosts ship an older one; if so and
# pysqlite3-binary is installed, use it instead (the same swap Chroma does on
# Colab). This runs before any edupulse module imports chromadb.
if sqlite3.sqlite_version_info < (3, 35, 0):  # pragma: no cover - depends on the host
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass  # chromadb will raise a clear error explaining the problem
