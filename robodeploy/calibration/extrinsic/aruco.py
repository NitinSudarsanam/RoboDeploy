"""ArUco marker extrinsic detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from robodeploy.calibration.base import CameraIntrinsics
from robodeploy.core.types import Pose3D


def _require_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for ArUco calibration") from exc


_ARUCO_DICTS: dict[str, str] = {
    "DICT_4X4_50": "DICT_4X4_50",
    "DICT_5X5_50": "DICT_5X5_50",
    "DICT_6X6_50": "DICT_6X6_50",
}


def _marker_object_points(marker_size_m: float) -> np.ndarray:
    half = float(marker_size_m) / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def _estimate_marker_poses(
    cv2: Any,
    corners: np.ndarray,
    *,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-marker r/t vectors (N,1,3), compatible with legacy OpenCV API."""
    estimate = getattr(cv2.aruco, "estimatePoseSingleMarkers", None)
    if estimate is not None:
        rvecs, tvecs, _ = estimate(corners, marker_size_m, camera_matrix, dist_coeffs)
        return rvecs, tvecs

    obj_pts = _marker_object_points(marker_size_m)
    rvecs_list: list[np.ndarray] = []
    tvecs_list: list[np.ndarray] = []
    for corner in corners:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, corner, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        rvecs_list.append(rvec.reshape(1, 1, 3))
        tvecs_list.append(tvec.reshape(1, 1, 3))
    if not rvecs_list:
        return np.zeros((0, 1, 3), dtype=np.float64), np.zeros((0, 1, 3), dtype=np.float64)
    return np.concatenate(rvecs_list, axis=0), np.concatenate(tvecs_list, axis=0)


class ArUcoExtrinsicCalibrator:
    """Detect ArUco markers and estimate marker poses in the camera frame."""

    def __init__(self, *, dictionary: str = "DICT_4X4_50", marker_size_m: float = 0.05) -> None:
        self.dictionary = dictionary
        self.marker_size_m = float(marker_size_m)

    def _get_dict(self):
        cv2 = _require_cv2()
        name = _ARUCO_DICTS.get(self.dictionary, self.dictionary)
        attr = getattr(cv2.aruco, name, None)
        if attr is None:
            raise ValueError(f"unknown ArUco dictionary: {self.dictionary}")
        return cv2.aruco.getPredefinedDictionary(attr)

    def fit(self, frames: list[np.ndarray], intrinsics: CameraIntrinsics) -> dict[int, Pose3D]:
        cv2 = _require_cv2()
        aruco_dict = self._get_dict()
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        K = np.asarray(intrinsics.to_matrix(), dtype=np.float64)
        dist = np.asarray(intrinsics.dist_coeffs, dtype=np.float64)
        out: dict[int, Pose3D] = {}
        for frame in frames:
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None or len(ids) == 0:
                continue
            rvecs, tvecs = _estimate_marker_poses(
                cv2,
                corners,
                marker_size_m=self.marker_size_m,
                camera_matrix=K,
                dist_coeffs=dist,
            )
            for idx, marker_id in enumerate(ids.reshape(-1)):
                mid = int(marker_id)
                t = tvecs[idx, 0]
                r = rvecs[idx, 0]
                R, _ = cv2.Rodrigues(r)
                from robodeploy.calibration.extrinsic.checkerboard import _rotation_matrix_to_quat_wxyz

                quat = _rotation_matrix_to_quat_wxyz(R)
                out[mid] = Pose3D(
                    position=(float(t[0]), float(t[1]), float(t[2])),
                    orientation=quat,
                )
        if not out:
            raise ValueError("no ArUco markers detected in frames")
        return out
