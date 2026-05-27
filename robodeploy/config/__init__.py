"""Lightweight preset loading without requiring Hydra."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PRESETS_PATH = Path(__file__).with_name("presets.yaml")


@lru_cache(maxsize=1)
def _load_all_presets() -> dict[str, dict[str, Any]]:
    if not _PRESETS_PATH.exists():
        return {}
    data = yaml.safe_load(_PRESETS_PATH.read_text(encoding="utf-8")) or {}
    return {str(name): dict(values) for name, values in data.items()}


def load_preset(name: str) -> dict[str, Any]:
    presets = _load_all_presets()
    if name not in presets:
        known = ", ".join(sorted(presets)) or "(none)"
        raise KeyError(f"Unknown preset '{name}'. Known presets: {known}")
    return dict(presets[name])
