"""RGB color-blob centroid → rough 3D object estimate in ``obs.objects``."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from robodeploy.core.transforms import ITransform
from robodeploy.core.types import Observation


def _quat_rotate_wxyz(quat: tuple[float, float, float, float], vec: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate a vector by quaternion (w, x, y, z)."""
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    vx, vy, vz = (float(vec[0]), float(vec[1]), float(vec[2]))
    ix = w * vx + y * vz - z * vy
    iy = w * vy + z * vx - x * vz
    iz = w * vz + x * vy - y * vx
    iw = -x * vx - y * vy - z * vz
    rx = ix * w + iw * -x + iy * -z - iz * -y
    ry = iy * w + iw * -y + iz * -x - ix * -z
    rz = iz * w + iw * -z + ix * -y - iy * -x
    return (rx, ry, rz)


def _camera_to_world(
    point_cam: tuple[float, float, float],
    extr: dict[str, object] | None,
    *,
    fallback_origin: tuple[float, float, float],
    fallback_scale: tuple[float, float, float],
    default_z: float,
) -> tuple[float, float, float]:
    if isinstance(extr, dict) and extr.get("position") and extr.get("orientation"):
        pos = extr["position"]
        quat = extr["orientation"]
        origin = (float(pos[0]), float(pos[1]), float(pos[2]))
        rotated = _quat_rotate_wxyz(
            (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            point_cam,
        )
        return (origin[0] + rotated[0], origin[1] + rotated[1], origin[2] + rotated[2])
    ox, oy, oz = fallback_origin
    sx, sy, sz = fallback_scale
    x_cam, y_cam, z_cam = point_cam
    return (ox + x_cam * sx, oy + y_cam * sy, oz + (z_cam - default_z) * sz)


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
        world_origin: tuple[float, float, float] = (0.55, 0.0, 0.38),
        camera_to_world_scale: tuple[float, float, float] = (0.15, 0.15, 1.0),
    ) -> None:
        self._camera = str(camera)
        self._object_name = str(object_name)
        self._target = np.asarray(target_rgb, dtype=np.int16)
        self._tolerance = int(tolerance)
        self._default_z = float(default_z)
        self._world_origin = tuple(float(v) for v in world_origin)
        self._world_scale = tuple(float(v) for v in camera_to_world_scale)

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

        x_cam = (cx - icx) * z / max(fx, 1e-6)
        y_cam = (cy - icy) * z / max(fy, 1e-6)
        z_cam = z

        extr = (getattr(obs, "camera_extrinsics", {}) or {}).get(self._camera)
        pos = _camera_to_world(
            (x_cam, y_cam, z_cam),
            extr if isinstance(extr, dict) else None,
            fallback_origin=self._world_origin,
            fallback_scale=self._world_scale,
            default_z=self._default_z,
        )
        objects = dict(getattr(obs, "objects", {}) or {})
        objects[self._object_name] = (pos, (1.0, 0.0, 0.0, 0.0))
        return replace(obs, objects=objects)
