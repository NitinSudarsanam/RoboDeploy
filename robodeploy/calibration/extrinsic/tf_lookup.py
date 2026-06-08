"""ROS TF extrinsic lookup (refactored from backends)."""

from __future__ import annotations

from robodeploy.backends.real.ros2.sensors.tf_extrinsics import (
    camera_info_to_intrinsics,
    compose_extrinsics,
    extrinsics_dict,
    lookup_camera_extrinsics,
    quat_multiply_wxyz,
    quat_rotate_wxyz,
)
from robodeploy.calibration.base import CameraIntrinsics
from robodeploy.core.types import Pose3D


class TfExtrinsicLookup:
    """Live TF lookup wrapper for calibration store integration."""

    def lookup(
        self,
        tf_buffer: object,
        *,
        target_frame: str,
        source_frame: str,
        timeout_s: float = 2.0,
        stamp: object | None = None,
    ) -> Pose3D | None:
        del timeout_s
        result = lookup_camera_extrinsics(
            tf_buffer,
            target_frame,
            source_frame,
            stamp=stamp,
        )
        if result is None:
            return None
        pos = result.get("position")
        quat = result.get("orientation")
        if pos is None or quat is None:
            return None
        return Pose3D(position=tuple(pos), orientation=tuple(quat))

    def intrinsics_from_camera_info(self, msg: object) -> CameraIntrinsics | None:
        raw = camera_info_to_intrinsics(msg)
        if raw is None:
            return None
        return CameraIntrinsics(
            fx=float(raw["fx"]),
            fy=float(raw["fy"]),
            cx=float(raw["cx"]),
            cy=float(raw["cy"]),
        )

    def to_store_payload(self, pose: Pose3D, *, frame_id: str, parent_link: str | None = None) -> dict:
        return extrinsics_dict(
            pose.position,
            pose.orientation,
            frame_id=frame_id,
            parent_link=parent_link,
            source="tf_lookup",
        )


__all__ = [
    "TfExtrinsicLookup",
    "camera_info_to_intrinsics",
    "compose_extrinsics",
    "extrinsics_dict",
    "lookup_camera_extrinsics",
    "quat_multiply_wxyz",
    "quat_rotate_wxyz",
]
