"""Optional perception sources for ROS2RealBackend prop pose queries."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from robodeploy.core.types import Observation


@runtime_checkable
class PerceptionSource(Protocol):
    def get_pose(self, prop_name: str) -> tuple[np.ndarray, np.ndarray]: ...


class DictPerceptionSource:
    """Static pose table for tests and mocap integrations."""

    def __init__(self, poses: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]) -> None:
        self._poses = dict(poses)

    def get_pose(self, prop_name: str) -> tuple[np.ndarray, np.ndarray]:
        if prop_name not in self._poses:
            raise KeyError(f"Perception source has no pose for prop '{prop_name}'.")
        pos, quat = self._poses[prop_name]
        return (
            np.asarray(pos, dtype=np.float64),
            np.asarray(quat, dtype=np.float64),
        )


class ScenePropPerceptionSource:
    """Mutable scene-oracle poses for ros2_rviz fake-sim pick demos (carry assist)."""

    def __init__(
        self,
        poses: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]],
    ) -> None:
        self._poses = poses

    def get_pose(self, prop_name: str) -> tuple[np.ndarray, np.ndarray]:
        if prop_name not in self._poses:
            raise KeyError(f"Scene oracle has no pose for prop '{prop_name}'.")
        pos, quat = self._poses[prop_name]
        return (
            np.asarray(pos, dtype=np.float64),
            np.asarray(quat, dtype=np.float64),
        )


class TFPerceptionSource:
    """Lookup prop poses from a TF buffer (world frame by default)."""

    def __init__(
        self,
        *,
        tf_buffer: Any,
        frame_by_prop: dict[str, str],
        target_frame: str = "world",
    ) -> None:
        self._tf_buffer = tf_buffer
        self._frame_by_prop = dict(frame_by_prop)
        self._target_frame = str(target_frame)

    def get_pose(self, prop_name: str) -> tuple[np.ndarray, np.ndarray]:
        frame = self._frame_by_prop.get(prop_name)
        if not frame:
            raise KeyError(f"TF perception source has no frame mapping for prop '{prop_name}'.")
        try:
            import rclpy.time

            query_time = rclpy.time.Time()
            tf_stamped = self._tf_buffer.lookup_transform(self._target_frame, frame, query_time)
        except Exception as exc:
            raise KeyError(f"TF lookup failed for prop '{prop_name}' frame '{frame}'.") from exc
        tr = tf_stamped.transform.translation
        rot = tf_stamped.transform.rotation
        return (
            np.asarray([float(tr.x), float(tr.y), float(tr.z)], dtype=np.float64),
            np.asarray([float(rot.w), float(rot.x), float(rot.y), float(rot.z)], dtype=np.float64),
        )


def _quat_rotate_wxyz(quat: tuple[float, float, float, float], vec: tuple[float, float, float]) -> tuple[float, float, float]:
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


class ColorBlobPerceptionSource:
    """Estimate prop pose from a color-blob centroid in a camera image."""

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
        self._latest_pose: tuple[np.ndarray, np.ndarray] | None = None

    def update_obs(self, obs: Observation) -> None:
        images = getattr(obs, "images", {}) or {}
        rgb = images.get(self._camera)
        if rgb is None:
            rgb = obs.rgb
        if rgb is None:
            return
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return
        diff = np.linalg.norm(arr[..., :3].astype(np.int16) - self._target, axis=-1)
        mask = diff <= self._tolerance
        if not np.any(mask):
            return
        ys, xs = np.nonzero(mask)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        depths = getattr(obs, "depths", {}) or {}
        depth_map = depths.get(self._camera) or obs.depth
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
        extr = (getattr(obs, "camera_extrinsics", {}) or {}).get(self._camera)
        if isinstance(extr, dict) and extr.get("position") and extr.get("orientation"):
            pos = extr["position"]
            quat = extr["orientation"]
            origin = (float(pos[0]), float(pos[1]), float(pos[2]))
            rotated = _quat_rotate_wxyz(
                (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                (x_cam, y_cam, z),
            )
            world_pos = (origin[0] + rotated[0], origin[1] + rotated[1], origin[2] + rotated[2])
        else:
            ox, oy, oz = self._world_origin
            sx, sy, sz = self._world_scale
            world_pos = (ox + x_cam * sx, oy + y_cam * sy, oz + (z - self._default_z) * sz)
        self._latest_pose = (
            np.asarray(world_pos, dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )

    def get_pose(self, prop_name: str) -> tuple[np.ndarray, np.ndarray]:
        if prop_name != self._object_name:
            raise KeyError(f"ColorBlobPerceptionSource tracks '{self._object_name}', not '{prop_name}'.")
        if self._latest_pose is None:
            raise KeyError(
                f"ColorBlobPerceptionSource has no blob detection yet for prop '{prop_name}'. "
                "Call update_obs() after each sensor merge."
            )
        return self._latest_pose
