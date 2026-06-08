"""Inject Gazebo camera / FT sensor links into a robot URDF from SensorRig mounts."""

from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

from robodeploy.core.types import SensorMount

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor


def _fmt(values) -> str:  # noqa: ANN001
    return " ".join(str(float(v)) for v in values)


def _quat_to_rpy(quat: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def _resolve_mount(sensor: "ISensor") -> SensorMount | None:
    mount = getattr(sensor, "mount", None)
    if isinstance(mount, SensorMount) and mount.parent_link:
        return mount
    cfg = dict(getattr(sensor, "config", {}) or {})
    raw = cfg.get("mount")
    if isinstance(raw, SensorMount) and raw.parent_link:
        return raw
    if isinstance(raw, dict) and raw.get("parent_link"):
        return SensorMount(**raw)
    return None


def _gz_sensor_topic(sensor_name: str, cfg: dict, kind: str) -> str:
    """Return the Gazebo transport topic matching ROS2 SensorRig defaults."""
    namespace = str(cfg.get("namespace", f"/{sensor_name}")).strip().rstrip("/")
    if not namespace.startswith("/"):
        namespace = f"/{namespace}"
    if kind == "camera":
        rel = str(cfg.get("rgb", "image_raw")).lstrip("/")
        return f"{namespace}/{rel}"
    rel = str(cfg.get("wrench_topic", cfg.get("topic", "wrench"))).lstrip("/")
    return f"{namespace}/{rel}"


def _sensor_kind(sensor: "ISensor") -> str:
    name = str(getattr(sensor, "name", "")).lower()
    cls = type(sensor).__name__.lower()
    if "camera" in cls or "camera" in name or "rgbd" in name:
        return "camera"
    if "ft" in cls or "ft" in name or "wrench" in name:
        return "ft"
    return "unknown"


def inject_sensors_into_urdf(urdf_text: str, sensors: list["ISensor"]) -> str:
    """Return URDF XML with fixed sensor links and Gazebo sensor blocks appended."""
    root = ET.fromstring(urdf_text)
    robot = root if root.tag == "robot" else root.find("robot")
    if robot is None:
        raise ValueError("URDF must contain a <robot> element.")

    for sensor in sensors:
        kind = _sensor_kind(sensor)
        if kind not in ("camera", "ft"):
            continue
        mount = _resolve_mount(sensor)
        if mount is None or not mount.parent_link:
            continue
        sensor_name = str(getattr(sensor, "name", "sensor"))
        link_name = f"{sensor_name}_link"
        joint_name = f"{sensor_name}_joint"
        parent = str(mount.parent_link).split("/")[-1]
        rpy = _quat_to_rpy(mount.orientation)

        joint = ET.SubElement(robot, "joint", {"name": joint_name, "type": "fixed"})
        origin = ET.SubElement(joint, "origin")
        origin.attrib["xyz"] = _fmt(mount.position)
        origin.attrib["rpy"] = _fmt(rpy)
        parent_elem = ET.SubElement(joint, "parent")
        parent_elem.attrib["link"] = parent
        child_elem = ET.SubElement(joint, "child")
        child_elem.attrib["link"] = link_name
        ET.SubElement(robot, "link", {"name": link_name})

        cfg = dict(getattr(sensor, "config", {}) or {})
        gazebo = ET.SubElement(robot, "gazebo", {"reference": link_name})
        gz_sensor = ET.SubElement(
            gazebo,
            "sensor",
            {"name": sensor_name, "type": "camera" if kind == "camera" else "force_torque"},
        )
        ET.SubElement(gz_sensor, "always_on").text = "true"
        ET.SubElement(gz_sensor, "update_rate").text = "30"
        topic_elem = ET.SubElement(gz_sensor, "topic")
        topic_elem.text = _gz_sensor_topic(sensor_name, cfg, kind)
        if kind == "camera":
            width = int(cfg.get("width", cfg.get("image_width", 640)))
            height = int(cfg.get("height", cfg.get("image_height", 480)))
            fovy = float(cfg.get("fovy_deg", 60.0))
            cam = ET.SubElement(gz_sensor, "camera")
            ET.SubElement(cam, "horizontal_fov").text = str(math.radians(fovy))
            image = ET.SubElement(cam, "image")
            ET.SubElement(image, "width").text = str(width)
            ET.SubElement(image, "height").text = str(height)
            ET.SubElement(image, "format").text = "R8G8B8"
            clip = ET.SubElement(cam, "clip")
            ET.SubElement(clip, "near").text = "0.01"
            ET.SubElement(clip, "far").text = "10.0"

    return ET.tostring(robot, encoding="unicode")


def write_urdf_with_sensors(urdf_path: str | Path, sensors: list["ISensor"]) -> Path:
    """Patch a URDF file with sensor links and return the temp output path."""
    text = Path(urdf_path).read_text(encoding="utf-8")
    patched = inject_sensors_into_urdf(text, sensors)
    fd, out = tempfile.mkstemp(prefix="robodeploy_gazebo_robot_", suffix=".urdf")
    Path(out).write_text(patched, encoding="utf-8")
    try:
        import os

        os.close(fd)
    except Exception:
        pass
    return Path(out)
