"""Example preset YAML — lives outside the robodeploy package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PRESETS_FILE = Path(__file__).with_name("presets.yaml")


def list_example_presets() -> list[str]:
    from robodeploy.config import list_presets

    return list_presets(presets_file=PRESETS_FILE)


def load_example_preset(name: str) -> dict[str, Any]:
    from robodeploy.config import load_preset

    return load_preset(name, presets_file=PRESETS_FILE)
