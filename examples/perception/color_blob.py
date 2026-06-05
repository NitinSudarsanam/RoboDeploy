"""RGB color-blob centroid → rough 3D object estimate in ``obs.objects``."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from robodeploy.core.transforms import ITransform
from robodeploy.core.types import Observation


class ColorBlobCentroidTransform(ITransform):
    """Estimate an object pose from a colored region in a wrist/overhead camera."""

    def __init__(
        self,
        *,
        camera: str = "wrist_camera",
        object_name: str = "source",
        target_rgb: tuple[int, int, int] = (255, 0, 0),
        tolerance: int = 90,
        default_z: float = 0.38,
    ) -> None:
        self._camera = str(camera)
        self._object_name = str(object_name)
        self._target = np.asarray(target_rgb, dtype=np.int16)
        self._tolerance = int(tolerance)
        self._default_z = float(default_z)

    def forward(self, obs: Observation) -> Observation:
        images = getattr(obs, "images", {}) or {}
        rgb = images.get(self._camera)
        if rgb is None:
            rgb = obs.rgb
        if rgb is None:
            return obs

        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return obs

        diff = np.linalg.norm(arr[..., :3].astype(np.int16) - self._target, axis=-1)
        mask = diff <= self._tolerance
        if not np.any(mask):
            return obs

        ys, xs = np.nonzero(mask)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        depths = getattr(obs, "depths", {}) or {}
        depth_map = depths.get(self._camera)
        if depth_map is None:
            depth_map = obs.depth

        z = self._default_z
        if depth_map is not None:
            darr = np.asarray(depth_map)
            if darr.ndim == 2:
                z = float(np.median(darr[ys, xs]))
            elif darr.ndim == 3:
                z = float(np.median(darr[ys, xs, 0]))

        intr = (getattr(obs, "camera_intrinsics", {}) or {}).get(self._camera, {})
        fx = float(intr.get("fx", arr.shape[1]))
        fy = float(intr.get("fy", arr.shape[0]))
        icx = float(intr.get("cx", arr.shape[1] * 0.5))
        icy = float(intr.get("cy", arr.shape[0] * 0.5))

        x = (cx - icx) * z / max(fx, 1e-6)
        y = (cy - icy) * z / max(fy, 1e-6)

        # Camera frame → simple tabletop frame (demo heuristic).
        pos = (0.55 + x * 0.15, y * 0.15, z)
        objects = dict(getattr(obs, "objects", {}) or {})
        objects[self._object_name] = (pos, (1.0, 0.0, 0.0, 0.0))
        return replace(obs, objects=objects)
