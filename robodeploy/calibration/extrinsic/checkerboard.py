"""Checkerboard-based camera extrinsic calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robodeploy.calibration.base import CameraIntrinsics
from robodeploy.calibration.store import CalibrationStore
from robodeploy.core.types import Pose3D

if TYPE_CHECKING:
    from robodeploy.env import RoboEnv


@dataclass
class CheckerboardSample:
    """One detected board pose in the camera frame."""

    image: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    corners: np.ndarray


def _require_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for checkerboard calibration") from exc


class CheckerboardExtrinsicCalibrator:
    """Detect checkerboard corners in N frames; solve PnP for camera pose."""

    def __init__(self, *, board_size: tuple[int, int] = (7, 5), square_size_m: float = 0.025) -> None:
        self.board_size = (int(board_size[0]), int(board_size[1]))
        self.square_size_m = float(square_size_m)
        self._object_points = self._make_object_points()

    def _make_object_points(self) -> np.ndarray:
        cols, rows = self.board_size
        objp = np.zeros((cols * rows, 3), dtype=np.float32)
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp[:, :2] = grid.astype(np.float32)
        objp *= float(self.square_size_m)
        return objp

    def detect(self, image: np.ndarray, intrinsics: CameraIntrinsics) -> CheckerboardSample | None:
        cv2 = _require_cv2()
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        if not found:
            return None
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        K = np.asarray(intrinsics.to_matrix(), dtype=np.float64)
        dist = np.asarray(intrinsics.dist_coeffs, dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(self._object_points, corners, K, dist)
        if not ok:
            return None
        return CheckerboardSample(
            image=image,
            rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
            tvec=np.asarray(tvec, dtype=np.float64).reshape(3),
            corners=np.asarray(corners, dtype=np.float64),
        )

    def capture(
        self,
        env: "RoboEnv",
        n_frames: int = 20,
        *,
        prompt_each_pose: bool = True,
        camera_key: str = "rgb",
        intrinsics: CameraIntrinsics | None = None,
    ) -> list[CheckerboardSample]:
        """Collect checkerboard samples from env observations."""
        samples: list[CheckerboardSample] = []
        default_intrinsics = intrinsics or CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        for i in range(int(n_frames)):
            if prompt_each_pose:
                pass  # caller may pause / reposition between frames
            obs = env.get_processed_obs_by_robot()
            primary = env.primary_robot
            frame = obs[primary.robot_id]
            image = getattr(frame, camera_key, None)
            if image is None and getattr(frame, "images", None):
                images = frame.images
                if images:
                    image = next(iter(images.values()))
            if image is None:
                continue
            img = np.asarray(image)
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img[:, :, ::-1]  # RGB → BGR for OpenCV
            sample = self.detect(img, default_intrinsics)
            if sample is not None:
                samples.append(sample)
            if len(samples) >= n_frames:
                break
            env.step(None)
        if len(samples) < 3:
            raise ValueError(f"need at least 3 checkerboard detections, got {len(samples)}")
        return samples

    def fit(self, samples: list[CheckerboardSample], intrinsics: CameraIntrinsics) -> Pose3D:
        if len(samples) < 3:
            raise ValueError("fit requires at least 3 samples")
        tvecs = np.stack([s.tvec for s in samples])
        mean_t = tvecs.mean(axis=0)
        # Average rotation via first sample rvec (full solve would use multi-view BA)
        rvec = samples[0].rvec
        cv2 = _require_cv2()
        R, _ = cv2.Rodrigues(rvec)
        # Convert R,t to position + quaternion wxyz
        quat = _rotation_matrix_to_quat_wxyz(R)
        return Pose3D(
            position=(float(mean_t[0]), float(mean_t[1]), float(mean_t[2])),
            orientation=quat,
        )

    def save(self, store: CalibrationStore, *, name: str, robot_id: str, pose: Pose3D) -> Any:
        return store.save(
            name,
            {
                "type": "checkerboard_extrinsic",
                "board_size": self.board_size,
                "square_size_m": self.square_size_m,
                "position": pose.position,
                "orientation": pose.orientation,
            },
            robot_id=robot_id,
        )


def _rotation_matrix_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(R, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
