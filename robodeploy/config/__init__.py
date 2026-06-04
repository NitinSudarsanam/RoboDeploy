"""Optional YAML preset loading (no bundled presets in the robodeploy package).

Presets and demo policies belong under ``examples/``. Pass an explicit ``presets_file``
path, or set ``ROBODEPLOY_PRESETS_FILE`` for the CLI.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REQUIRED_PRESET_KEYS = ("robot", "backend", "task", "policy")


@lru_cache(maxsize=16)
def _load_all_presets(presets_file: str) -> dict[str, dict[str, Any]]:
    path = Path(presets_file)
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(name): dict(values) for name, values in data.items()}


def list_presets(*, presets_file: Path | str) -> list[str]:
    return sorted(_load_all_presets(str(presets_file)).keys())


def load_preset(name: str, *, presets_file: Path | str) -> dict[str, Any]:
    presets = _load_all_presets(str(presets_file))
    if name not in presets:
        known = ", ".join(sorted(presets)) or "(none)"
        raise KeyError(f"Unknown preset '{name}' in {presets_file}. Known presets: {known}")
    preset = dict(presets[name])
    missing = [key for key in _REQUIRED_PRESET_KEYS if key not in preset]
    if missing:
        raise ValueError(f"Preset '{name}' missing required keys: {missing}")
    return preset
