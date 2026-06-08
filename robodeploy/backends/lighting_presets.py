"""Lighting preset library for SceneBuilder and SceneSpec."""

from __future__ import annotations

from robodeploy.core.types import LightSpec

_PRESETS: dict[str, list[LightSpec]] = {
    "minimal": [
        LightSpec(
            position=(0.0, 0.0, 2.0),
            direction=(0.0, 0.0, -1.0),
            diffuse=(0.6, 0.6, 0.6),
            kind="directional",
        ),
    ],
    "bright": [
        LightSpec(
            position=(1.0, -1.0, 3.0),
            direction=(-0.3, 0.3, -1.0),
            diffuse=(1.0, 1.0, 1.0),
            kind="directional",
        ),
        LightSpec(
            position=(-1.5, 1.0, 2.5),
            direction=(0.5, -0.2, -1.0),
            diffuse=(0.5, 0.5, 0.55),
            kind="directional",
        ),
    ],
    "dark": [
        LightSpec(
            position=(0.0, 0.0, 1.5),
            direction=(0.0, 0.0, -1.0),
            diffuse=(0.25, 0.25, 0.3),
            kind="directional",
        ),
    ],
    "randomized": [
        LightSpec(
            position=(0.5, -0.8, 2.2),
            direction=(0.0, 0.0, -1.0),
            diffuse=(0.7, 0.65, 0.6),
            kind="directional",
        ),
    ],
}


def get_lighting_preset(name: str) -> list[LightSpec]:
    key = str(name).lower()
    if key not in _PRESETS:
        raise KeyError(f"Unknown lighting preset '{name}'. Choose from: {', '.join(_PRESETS)}")
    return list(_PRESETS[key])


def lighting_preset_names() -> list[str]:
    return list(_PRESETS.keys())
