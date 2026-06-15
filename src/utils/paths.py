"""Resolve repo-relative paths used in configs and CSV file lists."""

from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | os.PathLike[str] | None) -> str | None:
    """Expand env vars and resolve non-absolute paths against the repo root."""
    if path is None:
        return None

    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    if os.path.isabs(expanded):
        return expanded

    return str(get_repo_root() / expanded)
