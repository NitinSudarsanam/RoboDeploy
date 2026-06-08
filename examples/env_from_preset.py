"""Build a RoboEnv from an example YAML preset."""

from __future__ import annotations

from typing import Any

from robodeploy.env import RoboEnv

from examples.config import PRESETS_FILE, load_example_preset


def env_from_preset(name: str, **overrides: Any) -> "RoboEnv":
    """Load ``examples/config/presets.yaml`` and construct via ``RoboEnv.from_config``."""
    cfg = {**load_example_preset(name), **overrides}
    return RoboEnv.from_config(cfg)


def example_presets_file() -> str:
    return str(PRESETS_FILE)
