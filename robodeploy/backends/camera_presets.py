"""Camera preset library for SceneBuilder and SceneSpec."""

from __future__ import annotations

from robodeploy.core.types import CameraSpec

_PRESETS: dict[str, list[CameraSpec]] = {
    "overview": [
        CameraSpec(
            name="overview",
            position=(0.0, -1.4, 0.9),
            orientation=(0.92, 0.0, 0.38, 0.0),
            fov_deg=65.0,
            resolution=(640, 480),
        ),
    ],
    "tabletop": [
        CameraSpec(
            name="tabletop",
            position=(0.6, -0.5, 0.55),
            orientation=(0.85, -0.2, 0.4, 0.2),
            fov_deg=55.0,
            resolution=(640, 480),
        ),
    ],
    "overhead": [
        CameraSpec(
            name="overhead",
            position=(0.0, 0.0, 1.2),
            orientation=(0.7071, 0.7071, 0.0, 0.0),
            fov_deg=50.0,
            resolution=(640, 480),
        ),
    ],
    "wrist": [
        CameraSpec(
            name="wrist_cam",
            position=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
            fov_deg=70.0,
            resolution=(128, 128),
            parent_link="ee_link",
        ),
    ],
}


def get_camera_preset(name: str) -> list[CameraSpec]:
    key = str(name).lower()
    if key not in _PRESETS:
        raise KeyError(f"Unknown camera preset '{name}'. Choose from: {', '.join(_PRESETS)}")
    return list(_PRESETS[key])


def camera_preset_names() -> list[str]:
    return list(_PRESETS.keys())
