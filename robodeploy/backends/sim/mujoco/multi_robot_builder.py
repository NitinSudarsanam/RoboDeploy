"""MuJoCo multi-robot MJCF composition."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from robodeploy.backends.sim.mujoco.scene_builder import MjcfSceneBuilder, _fmt
from robodeploy.core.types import Pose3D, SceneSpec
from robodeploy.description.base import RobotDescription

_RENAME_ATTRS = (
    "name",
    "joint",
    "body1",
    "body2",
    "site",
    "camera",
    "mesh",
    "hfield",
    "material",
    "texture",
    "target",
    "objname",
    "tendon",
)


@dataclass
class RobotMjcfSlice:
    robot_id: str
    joint_names: list[str]
    ee_link_name: str
    home_qpos: list[float] = field(default_factory=list)


class MultiRobotMjcfBuilder:
    """Compose one MJCF with namespaced robot bodies and shared scene props."""

    ROBOT_ROOT_NAME = "robot0/base"
    SOURCE_PREFIX = "robot0"

    def __init__(self, scene: SceneSpec, *, config: dict | None = None) -> None:
        self.config = dict(config or {})
        self.scene = scene
        self.root = ET.Element("mujoco", {"model": "robodeploy_multi"})
        ET.SubElement(self.root, "compiler", {"angle": "radian", "autolimits": "true"})
        ET.SubElement(
            self.root,
            "option",
            {
                "timestep": str(self.config.get("timestep", 0.002)),
                "gravity": "0 0 -9.81",
            },
        )
        ET.SubElement(self.root, "default")
        self.robot_slices: dict[str, RobotMjcfSlice] = {}

    def add_robot(
        self,
        robot_id: str,
        description: RobotDescription,
        mjcf_xml: str,
        *,
        base_pose: Pose3D | None = None,
        meshdir: str | None = None,
    ) -> RobotMjcfSlice:
        if robot_id in self.robot_slices:
            raise ValueError(f"Duplicate robot_id '{robot_id}' in MultiRobotMjcfBuilder.")

        pose = base_pose or Pose3D()
        src_root = ET.fromstring(mjcf_xml)
        if meshdir:
            compiler = src_root.find("compiler")
            if compiler is not None and "meshdir" in compiler.attrib:
                dst_compiler = self._child("compiler")
                dst_compiler.attrib.setdefault("meshdir", str(meshdir).replace("\\", "/"))

        src_wb = src_root.find("worldbody")
        if src_wb is None:
            raise ValueError("Robot MJCF is missing <worldbody>.")

        robot_body = None
        for child in list(src_wb):
            if child.tag == "body" and child.attrib.get("name") == self.ROBOT_ROOT_NAME:
                robot_body = copy.deepcopy(child)
                break
        if robot_body is None:
            raise KeyError(
                f"Robot root body '{self.ROBOT_ROOT_NAME}' not found. "
                "Bundled demo MJCF uses robot0/base as the arm root."
            )

        self._rename_prefix(robot_body, self.SOURCE_PREFIX, robot_id)
        self._namespace_local_names(robot_body, robot_id)
        robot_body.attrib["pos"] = _fmt(pose.position)
        if pose.orientation != (1.0, 0.0, 0.0, 0.0):
            robot_body.attrib["quat"] = _fmt(pose.orientation)

        dst_wb = self._child("worldbody")
        dst_wb.append(robot_body)

        src_act = src_root.find("actuator")
        if src_act is not None:
            dst_act = self._child("actuator")
            for elem in list(src_act):
                cloned = copy.deepcopy(elem)
                self._rename_prefix(cloned, self.SOURCE_PREFIX, robot_id)
                self._namespace_local_names(cloned, robot_id)
                dst_act.append(cloned)

        joint_names = [str(jn).replace(self.SOURCE_PREFIX, robot_id) for jn in description.joint_names]
        ee_link = str(description.ee_link_name).replace(self.SOURCE_PREFIX, robot_id)
        home = [float(v) for v in getattr(description, "home_qpos", [])]
        sl = RobotMjcfSlice(
            robot_id=robot_id,
            joint_names=joint_names,
            ee_link_name=ee_link,
            home_qpos=home,
        )
        self.robot_slices[robot_id] = sl
        return sl

    def finalize(self, sensors: list) -> str:
        builder = MjcfSceneBuilder(ET.tostring(self.root, encoding="unicode"), config=self.config)
        builder.ensure_world_defaults(add_camera=not bool(self.scene.to_world().cameras))
        builder.attach_world(self.scene.to_world())
        builder.attach_sensors(sensors)
        return builder.emit()

    def _child(self, tag: str) -> ET.Element:
        elem = self.root.find(tag)
        if elem is None:
            elem = ET.SubElement(self.root, tag)
        return elem

    @classmethod
    def _rename_prefix(cls, elem: ET.Element, old: str, new: str) -> None:
        for attr in _RENAME_ATTRS:
            if attr in elem.attrib and old in elem.attrib[attr]:
                elem.attrib[attr] = elem.attrib[attr].replace(old, new)
        for child in list(elem):
            cls._rename_prefix(child, old, new)

    @classmethod
    def _namespace_local_names(cls, elem: ET.Element, robot_id: str) -> None:
        """Prefix bare MJCF names (e.g. eye_in_hand) to avoid cross-robot collisions."""
        prefix = f"{robot_id}/"
        for node in elem.iter():
            for attr in _RENAME_ATTRS:
                val = node.attrib.get(attr)
                if not val or val.startswith(prefix) or "/" in val:
                    continue
                node.attrib[attr] = prefix + val


def load_robot_mjcf_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")
