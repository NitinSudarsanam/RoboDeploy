"""TF-tree helpers for camera extrinsics on ROS2 sensor streams."""

from __future__ import annotations

from typing import Any, Optional


def quat_multiply_wxyz(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product (w, x, y, z)."""
    aw, ax, ay, az = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    bw, bx, by, bz = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_rotate_wxyz(
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


def compose_extrinsics(
    parent_pos: tuple[float, float, float],
    parent_quat: tuple[float, float, float, float],
    child_pos: tuple[float, float, float],
    child_quat: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Compose parent_T_child into world pose for the child frame."""
    rotated = quat_rotate_wxyz(parent_quat, child_pos)
    pos = (
        float(parent_pos[0]) + rotated[0],
        float(parent_pos[1]) + rotated[1],
        float(parent_pos[2]) + rotated[2],
    )
    quat = quat_multiply_wxyz(parent_quat, child_quat)
    return pos, quat


def extrinsics_dict(
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
    *,
    frame_id: str,
    parent_link: str | None = None,
    source: str = "tf",
) -> dict[str, object]:
    out: dict[str, object] = {
        "position": position,
        "orientation": orientation,
        "frame_id": str(frame_id),
        "source": str(source),
    }
    if parent_link:
        out["parent_link"] = str(parent_link)
    return out


def camera_info_to_intrinsics(msg: Any) -> dict[str, float] | None:
    """Extract fx/fy/cx/cy from sensor_msgs/CameraInfo."""
    if msg is None:
        return None
    k = getattr(msg, "k", None)
    if k is None or len(k) < 9:
        return None
    width = float(getattr(msg, "width", 0) or 0)
    height = float(getattr(msg, "height", 0) or 0)
    return {
        "width": width,
        "height": height,
        "fx": float(k[0]),
        "fy": float(k[4]),
        "cx": float(k[2]),
        "cy": float(k[5]),
    }


def lookup_camera_extrinsics(
    tf_buffer: Any,
    target_frame: str,
    camera_frame: str,
    *,
    stamp: Any = None,
) -> dict[str, object] | None:
    """Lookup target_T_camera and return Observation-compatible extrinsics dict."""
    if tf_buffer is None:
        return None
    target = str(target_frame or "").strip()
    camera = str(camera_frame or "").strip()
    if not target or not camera:
        return None
    try:
        query_time = stamp
        if query_time is None:
            import rclpy.time

            query_time = rclpy.time.Time()
        tf_stamped = tf_buffer.lookup_transform(target, camera, query_time)
    except Exception:
        return None
    tr = tf_stamped.transform.translation
    rot = tf_stamped.transform.rotation
    child = getattr(tf_stamped, "child_frame_id", None)
    frame_id = str(child).strip() if child else camera
    return extrinsics_dict(
        (float(tr.x), float(tr.y), float(tr.z)),
        (float(rot.w), float(rot.x), float(rot.y), float(rot.z)),
        frame_id=frame_id,
        parent_link=target,
        source="tf",
    )
