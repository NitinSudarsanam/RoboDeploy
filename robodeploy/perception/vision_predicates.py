"""Classical vision predicates — color blobs, ArUco markers, learned pose injection."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from robodeploy.core.transforms import ITransform
from robodeploy.core.types import Observation

Pose3D = tuple[tuple[float, float, float], tuple[float, float, float, float]]


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image [H,W,3] to HSV float in OpenCV ranges."""
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = np.max(arr, axis=-1)
    cmin = np.min(arr, axis=-1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    mask = delta > 1e-8
    rmask = mask & (cmax == r)
    gmask = mask & (cmax == g)
    bmask = mask & (cmax == b)
    hue[rmask] = 60.0 * (((g[rmask] - b[rmask]) / delta[rmask]) % 6.0)
    hue[gmask] = 60.0 * (((b[gmask] - r[gmask]) / delta[gmask]) + 2.0)
    hue[bmask] = 60.0 * (((r[bmask] - g[bmask]) / delta[bmask]) + 4.0)
    hue = np.where(hue < 0.0, hue + 360.0, hue)

    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(cmax > 1e-8, delta / cmax, 0.0)
    val = cmax
    return np.stack([hue, sat * 255.0, val * 255.0], axis=-1)


def count_hsv_pixels(
    rgb: np.ndarray,
    *,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> int:
    """Count pixels inside an HSV range (OpenCV H:0-179, S/V:0-255)."""
    hsv = rgb_to_hsv(rgb)
    h = hsv[..., 0] * 0.5  # scale to OpenCV hue range
    s = hsv[..., 1]
    v = hsv[..., 2]
    lo = np.asarray(lower, dtype=np.float32)
    hi = np.asarray(upper, dtype=np.float32)
    mask = (h >= lo[0]) & (h <= hi[0]) & (s >= lo[1]) & (s <= hi[1]) & (v >= lo[2]) & (v <= hi[2])
    return int(np.sum(mask))


def _quat_rotate_wxyz(
    quat: tuple[float, float, float, float],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
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


def has_camera_extrinsics(extr: dict[str, object] | None) -> bool:
    """True when extrinsics carry both world position and orientation (wxyz)."""
    return (
        isinstance(extr, dict)
        and extr.get("position") is not None
        and extr.get("orientation") is not None
    )


def _camera_to_world(
    point_cam: tuple[float, float, float],
    extr: dict[str, object] | None,
    *,
    fallback_origin: tuple[float, float, float],
    fallback_scale: tuple[float, float, float],
    default_z: float,
) -> tuple[float, float, float]:
    """Map a camera-frame point to world coordinates.

    When ``extr`` has ``position`` and ``orientation`` (wxyz quaternion), the point is
    rotated by the camera orientation and translated by the camera origin — the path
    used with MuJoCo live mounts and ROS TF lookups.

    Without both fields, falls back to axis-aligned scaling from ``fallback_origin``
    using ``fallback_scale``. That heuristic guesses desk/workspace geometry and is
    **not** metrically correct; enable it only via ``fallback_mode=True`` on
    ``ColorBlobTracker`` / ``ColorBlobTrackerTransform``.
    """
    if has_camera_extrinsics(extr):
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


class ColorBlobTracker:
    """Detect a colored blob and unproject centroid to a 3D pose estimate."""

    def __init__(
        self,
        hsv_range: tuple[tuple[float, float, float], tuple[float, float, float]],
        *,
        min_pixels: int = 200,
        default_z: float = 0.38,
        world_origin: tuple[float, float, float] = (0.55, 0.0, 0.38),
        camera_to_world_scale: tuple[float, float, float] = (0.15, 0.15, 1.0),
        fallback_mode: bool = False,
    ) -> None:
        self._lower, self._upper = hsv_range
        self._min_pixels = int(min_pixels)
        self._default_z = float(default_z)
        self._world_origin = tuple(float(v) for v in world_origin)
        self._world_scale = tuple(float(v) for v in camera_to_world_scale)
        self._fallback_mode = bool(fallback_mode)

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        intrinsics: dict[str, float] | None,
        extrinsics: dict[str, object] | None,
    ) -> Pose3D | None:
        if rgb is None:
            return None
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return None
        if count_hsv_pixels(arr, lower=self._lower, upper=self._upper) < self._min_pixels:
            return None

        hsv = rgb_to_hsv(arr)
        h = hsv[..., 0] * 0.5
        s = hsv[..., 1]
        v = hsv[..., 2]
        lo = np.asarray(self._lower, dtype=np.float32)
        hi = np.asarray(self._upper, dtype=np.float32)
        mask = (h >= lo[0]) & (h <= hi[0]) & (s >= lo[1]) & (s <= hi[1]) & (v >= lo[2]) & (v <= hi[2])
        if not np.any(mask):
            return None

        ys, xs = np.nonzero(mask)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        z = self._default_z
        if depth is not None:
            darr = np.asarray(depth)
            if darr.ndim == 2:
                z = float(np.median(darr[ys, xs]))
            elif darr.ndim == 3:
                z = float(np.median(darr[ys, xs, 0]))

        intr = intrinsics or {}
        fx = float(intr.get("fx", arr.shape[1]))
        fy = float(intr.get("fy", arr.shape[0]))
        icx = float(intr.get("cx", arr.shape[1] * 0.5))
        icy = float(intr.get("cy", arr.shape[0] * 0.5))

        x_cam = (cx - icx) * z / max(fx, 1e-6)
        y_cam = (cy - icy) * z / max(fy, 1e-6)
        extr = extrinsics if isinstance(extrinsics, dict) else None
        if not has_camera_extrinsics(extr) and not self._fallback_mode:
            return None
        pos = _camera_to_world(
            (x_cam, y_cam, z),
            extr,
            fallback_origin=self._world_origin,
            fallback_scale=self._world_scale,
            default_z=self._default_z,
        )
        return (pos, (1.0, 0.0, 0.0, 0.0))


class ArUcoTracker:
    """OpenCV ArUco marker detection (optional opencv dependency)."""

    def __init__(self, *, marker_size_m: float = 0.04, dictionary: str = "DICT_4X4_50") -> None:
        self._marker_size_m = float(marker_size_m)
        self._dictionary = str(dictionary)

    def detect(
        self,
        rgb: np.ndarray,
        intrinsics: dict[str, float] | None,
        extrinsics: dict[str, object] | None,
    ) -> dict[int, Pose3D]:
        del extrinsics
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError:
            return {}

        arr = np.asarray(rgb)
        if arr.ndim != 3:
            return {}
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        dict_id = getattr(cv2.aruco, self._dictionary, cv2.aruco.DICT_4X4_50)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return {}

        intr = intrinsics or {}
        fx = float(intr.get("fx", arr.shape[1]))
        fy = float(intr.get("fy", arr.shape[0]))
        cx = float(intr.get("cx", arr.shape[1] * 0.5))
        cy = float(intr.get("cy", arr.shape[0] * 0.5))
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)

        out: dict[int, Pose3D] = {}
        for corner, marker_id in zip(corners, ids.flatten()):
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner, self._marker_size_m, camera_matrix, dist
            )
            t = tvecs[0][0]
            pos = (float(t[0]), float(t[1]), float(t[2]))
            # orientation omitted — identity quaternion placeholder
            out[int(marker_id)] = (pos, (1.0, 0.0, 0.0, 0.0))
        return out


class LearnedPoseEstimator:
    """Wrapper for user-injected nn.Module predicting object poses from RGB-D."""

    def __init__(
        self,
        model_fn: Callable[[np.ndarray, np.ndarray | None], dict[str, Pose3D]],
    ) -> None:
        self._model_fn = model_fn

    def estimate(self, rgb: np.ndarray, depth: np.ndarray | None) -> dict[str, Pose3D]:
        return self._model_fn(rgb, depth)


class ColorBlobTrackerTransform(ITransform):
    """ObsPipeline transform — populates ``obs.objects`` from ColorBlobTracker."""

    def __init__(
        self,
        *,
        camera: str = "wrist_camera",
        object_name: str = "source",
        hsv_range: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        target_rgb: tuple[int, int, int] = (255, 0, 0),
        tolerance: int = 90,
        min_pixels: int = 100,
        default_z: float = 0.38,
        world_origin: tuple[float, float, float] = (0.55, 0.0, 0.38),
        camera_to_world_scale: tuple[float, float, float] = (0.15, 0.15, 1.0),
        fallback_mode: bool = False,
    ) -> None:
        self._camera = str(camera)
        self._object_name = str(object_name)
        self._target_rgb = np.asarray(target_rgb, dtype=np.int16)
        self._tolerance = int(tolerance)
        tracker_kwargs = dict(
            min_pixels=min_pixels,
            default_z=default_z,
            world_origin=world_origin,
            camera_to_world_scale=camera_to_world_scale,
            fallback_mode=fallback_mode,
        )
        if hsv_range is not None:
            self._tracker = ColorBlobTracker(hsv_range, **tracker_kwargs)
            self._use_rgb = False
        else:
            self._tracker = ColorBlobTracker(
                ((0.0, 80.0, 80.0), (10.0, 255.0, 255.0)),
                **tracker_kwargs,
            )
            self._use_rgb = True

    def forward(self, obs: Observation) -> Observation:
        images = getattr(obs, "images", {}) or {}
        rgb = images.get(self._camera)
        if rgb is None:
            rgb = obs.rgb
        if rgb is None:
            return obs

        depths = getattr(obs, "depths", {}) or {}
        depth = depths.get(self._camera)
        if depth is None:
            depth = obs.depth

        intr = (getattr(obs, "camera_intrinsics", {}) or {}).get(self._camera, {})
        extr = (getattr(obs, "camera_extrinsics", {}) or {}).get(self._camera)

        arr = np.asarray(rgb)
        if self._use_rgb:
            diff = np.linalg.norm(arr[..., :3].astype(np.int16) - self._target_rgb, axis=-1)
            mask = diff <= self._tolerance
            if not np.any(mask):
                return obs
            ys, xs = np.nonzero(mask)
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            z = self._tracker._default_z
            if depth is not None:
                darr = np.asarray(depth)
                if darr.ndim == 2:
                    z = float(np.median(darr[ys, xs]))
                elif darr.ndim == 3:
                    z = float(np.median(darr[ys, xs, 0]))
            fx = float(intr.get("fx", arr.shape[1]))
            fy = float(intr.get("fy", arr.shape[0]))
            icx = float(intr.get("cx", arr.shape[1] * 0.5))
            icy = float(intr.get("cy", arr.shape[0] * 0.5))
            x_cam = (cx - icx) * z / max(fx, 1e-6)
            y_cam = (cy - icy) * z / max(fy, 1e-6)
            extr_dict = extr if isinstance(extr, dict) else None
            if not has_camera_extrinsics(extr_dict) and not self._tracker._fallback_mode:
                return obs
            pos = _camera_to_world(
                (x_cam, y_cam, z),
                extr_dict,
                fallback_origin=self._tracker._world_origin,
                fallback_scale=self._tracker._world_scale,
                default_z=self._tracker._default_z,
            )
            pose: Pose3D | None = (pos, (1.0, 0.0, 0.0, 0.0))
        else:
            pose = self._tracker.detect(rgb, depth, intr, extr)

        if pose is None:
            return obs

        objects = dict(getattr(obs, "objects", {}) or {})
        objects[self._object_name] = pose
        return replace(obs, objects=objects)
