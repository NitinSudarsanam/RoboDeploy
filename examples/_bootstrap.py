"""Shared bootstrap helpers for example scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_on_path() -> Path:
    """Insert repo root on sys.path when running examples without editable install."""
    repo_root = Path(__file__).resolve().parents[1]
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return repo_root
