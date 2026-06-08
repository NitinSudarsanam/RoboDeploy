"""Artifact helpers for MuJoCo showcase demos (montage PNG, JSON snapshot)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_rgb_image(path: Path, rgb: np.ndarray) -> bool:
    arr = np.asarray(rgb, dtype=np.uint8)
    try:
        from PIL import Image

        Image.fromarray(arr).save(path)
        return True
    except Exception:
        ppm_path = path.with_suffix(".ppm")
        h, w = arr.shape[:2]
        with ppm_path.open("wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(arr.tobytes())
        print(f"Wrote {ppm_path} (install Pillow for PNG)")
        return False


def montage(images: dict[str, np.ndarray]) -> np.ndarray:
    panels = [np.asarray(img, dtype=np.uint8) for img in images.values() if img is not None]
    if not panels:
        return np.zeros((48, 64, 3), dtype=np.uint8)
    h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != h:
            scale = h / float(p.shape[0])
            w = max(1, int(round(p.shape[1] * scale)))
            idx = (np.linspace(0, p.shape[0] - 1, h)).astype(int)
            jdx = (np.linspace(0, p.shape[1] - 1, w)).astype(int)
            p = p[np.ix_(idx, jdx)]
        resized.append(p)
    return np.concatenate(resized, axis=1)


def obs_snapshot(obs) -> dict[str, Any]:
    def _vec(v):
        if v is None:
            return None
        return [float(x) for x in np.asarray(v).reshape(-1)]

    return {
        "objects": list(getattr(obs, "objects", {}).keys()),
        "sensor_status": dict(getattr(obs, "sensor_status", {}) or {}),
        "camera_intrinsics": dict(getattr(obs, "camera_intrinsics", {}) or {}),
        "camera_extrinsics": dict(getattr(obs, "camera_extrinsics", {}) or {}),
        "ft_forces": {k: _vec(v) for k, v in (getattr(obs, "ft_forces", {}) or {}).items()},
        "ft_torques": {k: _vec(v) for k, v in (getattr(obs, "ft_torques", {}) or {}).items()},
        "imu_acceleration": _vec(getattr(obs, "imu_acceleration", None)),
        "imu_angular_velocity": _vec(getattr(obs, "imu_angular_velocity", None)),
    }


def write_showcase_artifacts(
    obs,
    scene_text: str,
    out_dir: Path,
    *,
    skip_render: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "showcase_montage.png"
    json_path = out_dir / "showcase_snapshot.json"
    txt_path = out_dir / "showcase_scene.txt"

    if not skip_render:
        montage_imgs = {}
        images = getattr(obs, "images", {}) or {}
        if images.get("wrist_camera") is not None:
            montage_imgs["wrist_camera"] = np.asarray(images["wrist_camera"])
        if images.get("overhead_camera") is not None:
            montage_imgs["overhead_camera"] = np.asarray(images["overhead_camera"])
        if montage_imgs:
            save_rgb_image(png_path, montage(montage_imgs))

    snapshot = obs_snapshot(obs)
    json_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    txt_path.write_text(scene_text + "\n\n" + json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    if png_path.exists():
        print(f"Wrote {png_path}")
