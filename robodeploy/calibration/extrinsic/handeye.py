"""Hand-eye calibration (Tsai-Lenz / Park-Martin / Daniilidis)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from robodeploy.core.types import Pose3D


def _require_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for hand-eye calibration") from exc


_METHOD_MAP: dict[str, int] = {
    "tsai": 0,
    "park": 1,
    "daniilidis": 2,
}


class HandEyeCalibrator:
    """Tsai-Lenz / Park-Martin / Daniilidis hand-eye calibration via OpenCV."""

    def fit(
        self,
        robot_poses: list[Pose3D],
        marker_poses: list[Pose3D],
        method: Literal["tsai", "park", "daniilidis"] = "park",
    ) -> Pose3D:
        if len(robot_poses) < 3 or len(marker_poses) < 3:
            raise ValueError("hand-eye calibration requires at least 3 pose pairs")
        if len(robot_poses) != len(marker_poses):
            raise ValueError("robot_poses and marker_poses must have equal length")
        cv2 = _require_cv2()
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        for rp, mp in zip(robot_poses, marker_poses):
            Rg, tg = _pose_to_rt(rp)
            Rt, tc = _pose_to_rt(mp)
            R_gripper2base.append(Rg)
            t_gripper2base.append(tg)
            R_target2cam.append(Rt)
            t_target2cam.append(tc)
        method_id = _METHOD_MAP.get(method, _METHOD_MAP["park"])
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=method_id,
        )
        from robodeploy.calibration.extrinsic.checkerboard import _rotation_matrix_to_quat_wxyz

        quat = _rotation_matrix_to_quat_wxyz(R_cam2gripper)
        t = np.asarray(t_cam2gripper, dtype=np.float64).reshape(3)
        return Pose3D(
            position=(float(t[0]), float(t[1]), float(t[2])),
            orientation=quat,
        )

    def rotation_diversity(self, robot_poses: list[Pose3D]) -> float:
        """Condition proxy: mean angle between successive orientations (rad)."""
        if len(robot_poses) < 2:
            return 0.0
        angles = []
        for i in range(1, len(robot_poses)):
            q0 = np.asarray(robot_poses[i - 1].orientation, dtype=np.float64)
            q1 = np.asarray(robot_poses[i].orientation, dtype=np.float64)
            dot = float(np.clip(np.abs(np.dot(q0, q1)), -1.0, 1.0))
            angles.append(2.0 * np.arccos(dot))
        return float(np.mean(angles))


def _pose_to_rt(pose: Pose3D) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _require_cv2()
    from robodeploy.calibration.extrinsic.checkerboard import _rotation_matrix_to_quat_wxyz

    w, x, y, z = pose.orientation
    # quaternion wxyz → rotation matrix
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    t = np.asarray(pose.position, dtype=np.float64).reshape(3, 1)
    return R, t
