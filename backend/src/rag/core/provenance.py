"""What produced a result: the config (and its hash) and the git commit of the code.

Stored with every eval run, so a number can always be traced back to the exact setup, and two
runs can be checked for "same setup?" by comparing hashes instead of reading configs.
"""

import hashlib
import json
import subprocess
from typing import Any

from rag.core.config import BACKEND_DIR, Settings

# Sections that change results. `database` is left out (it holds a password and doesn't change
# what the pipeline computes), so is `index` (the index version is recorded separately).
_RESULT_SECTIONS = {"dataset", "llm", "embeddings", "chunking", "retrieval", "prompts", "evaluation"}


def config_snapshot(settings: Settings) -> dict[str, Any]:
    return settings.model_dump(mode="json", include=_RESULT_SECTIONS)


def config_hash(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()[:12]


def git_state() -> dict[str, Any]:
    """Commit hash, and whether there were uncommitted changes (then the commit alone isn't enough)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=BACKEND_DIR, capture_output=True, text=True).stdout.strip()

    return {"git_commit": git("rev-parse", "--short", "HEAD") or None,
            "git_dirty": bool(git("status", "--porcelain"))}
