"""MuJoCo MJCF scene composition helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from robodeploy.core.spaces import AssetFormat
from robodeploy.core.types import GeomSpec, PropConfig, WorldSpec


def _fmt(values) -> str:  # noqa: ANN001
    return " ".join(str(float(v)) for v in values)


class MjcfSceneBuilder:
    """Build final MJCF by combining robot XML, actuators, and WorldSpec."""

    def __init__(self, robot_mjcf_xml: str, *, config: dict | None = None) -> None:
        self.config = dict(config or {})
        self.root = ET.fromstring(robot_mjcf_xml)
        if self.root.tag != "mujoco":
            raise ValueError("Expected MJCF root tag 'mujoco'.")

    @classmethod
    def from_compiled_model(cls, mujoco, model, *, config: dict | None = None):  # noqa: ANN001
        fd, tmp_path = tempfile.mkstemp(prefix="robodeploy_mj_", suffix=".xml")
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            mujoco.mj_saveLastXML(str(tmp_path), model)
            xml_text = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
            try:
                import os

                os.close(fd)
            except Exception:
                pass
        return cls(xml_text, config=config)

    def ensure_compiler_meshdir(self, meshdir: str | None) -> None:
        if not meshdir:
            return
        compiler = self._child("compiler")
        compiler.attrib.setdefault("meshdir", str(meshdir).replace("\\", "/"))

    def stabilize_urdf_import(self) -> None:
        opt = self._child("option")
        opt.attrib.setdefault("timestep", str(self.config.get("urdf_timestep", 0.001)))
        opt.attrib.setdefault("integrator", str(self.config.get("urdf_integrator", "RK4")))

        min_mass = float(self.config.get("urdf_min_mass", 0.01))
        min_inertia_diag = float(self.config.get("urdf_min_inertia_diag", 1e-4))
        for body in self.root.iter("body"):
            inertial = body.find("inertial")
            if inertial is None:
                continue
            if "mass" in inertial.attrib:
                try:
                    mass = float(inertial.attrib["mass"])
                except Exception:
                    mass = min_mass
                if mass < min_mass:
                    inertial.attrib["mass"] = str(min_mass)
            if "diaginertia" in inertial.attrib:
                parts = str(inertial.attrib.get("diaginertia", "")).split()
                if len(parts) == 3:
                    vals = []
                    for part in parts:
                        try:
                            val = float(part)
                        except Exception:
                            val = min_inertia_diag
                        vals.append(str(max(val, min_inertia_diag)))
                    inertial.attrib["diaginertia"] = " ".join(vals)

        damping = float(self.config.get("urdf_joint_damping", 1.0))
        armature = float(self.config.get("urdf_joint_armature", 0.01))
        for joint in self.root.iter("joint"):
            joint.attrib.setdefault("damping", str(damping))
            joint.attrib.setdefault("armature", str(armature))

    def ensure_world_defaults(self, *, add_camera: bool = True) -> None:
        worldbody = self._child("worldbody")
        if not any(child.tag == "light" for child in list(worldbody)):
            ET.SubElement(
                worldbody,
                "light",
                {"pos": "0 0 2.0", "dir": "0 0 -1", "diffuse": "0.8 0.8 0.8", "specular": "0.2 0.2 0.2"},
            )
        if not any(child.tag == "geom" and child.attrib.get("type") == "plane" for child in list(worldbody)):
            ET.SubElement(
                worldbody,
                "geom",
                {"name": "floor", "type": "plane", "size": "2 2 0.1", "rgba": "0.85 0.85 0.85 1", "pos": "0 0 0"},
            )
        if add_camera and not any(child.tag == "camera" for child in list(worldbody)):
            ET.SubElement(
                worldbody,
                "camera",
                {"name": "main", "pos": "0 -1.2 0.6", "xyaxes": "1 0 0 0 0 1"},
            )

    def attach_actuators(self, joint_names: list[str], *, kp: float | None = None) -> None:
        actuator = self._child("actuator")
        existing: set[str] = set()
        for elem in list(actuator):
            if elem.attrib.get("joint"):
                existing.add(str(elem.attrib["joint"]))
            if elem.attrib.get("name"):
                existing.add(str(elem.attrib["name"]))

        gain = float(kp if kp is not None else self.config.get("urdf_position_kp", 50.0))
        for joint_name in joint_names:
            if joint_name in existing:
                continue
            ET.SubElement(actuator, "position", {"name": str(joint_name), "joint": str(joint_name), "kp": str(gain)})

    def attach_world(self, world: WorldSpec) -> None:
        opt = self._child("option")
        opt.attrib["gravity"] = _fmt(world.gravity)
        self._attach_terrain(world)
        self._attach_lights(world)
        for prop in world.props:
            self._attach_prop(prop)
        self._attach_cameras(world)

    def attach_sensors(self, sensors: list) -> None:
        """Emit MuJoCo camera/site/sensor XML for mounted RoboDeploy sensors."""

        for sensor in sensors:
            cls_name = type(sensor).__name__.lower()
            name = str(getattr(sensor, "name", "sensor"))
            config = dict(getattr(sensor, "config", {}) or {})
            if bool(getattr(sensor, "is_real", False)):
                continue
            if "camera" in cls_name or "camera" in name.lower():
                self._attach_mounted_camera(sensor, config)
            elif "ft" in cls_name or "force" in cls_name or name.lower().endswith("_ft"):
                self._attach_mounted_ft_sensor(sensor, config)

    def emit(self) -> str:
        return ET.tostring(self.root, encoding="unicode")

    def _child(self, tag: str) -> ET.Element:
        elem = self.root.find(tag)
        if elem is None:
            elem = ET.SubElement(self.root, tag)
        return elem

    def _attach_terrain(self, world: WorldSpec) -> None:
        terrain = world.terrain
        worldbody = self._child("worldbody")
        plane = None
        for child in list(worldbody):
            if child.tag == "geom" and child.attrib.get("type") == "plane":
                plane = child
                break
        if terrain.kind == "flat":
            if plane is None:
                plane = ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane"})
            plane.attrib.update({"size": f"{float(terrain.size[0])} {float(terrain.size[1])} 0.1", "pos": "0 0 0"})
            plane.attrib.setdefault("rgba", "0.85 0.85 0.85 1")
            return

        if terrain.kind == "heightfield" and terrain.heightfield_path:
            asset = self._child("asset")
            ET.SubElement(
                asset,
                "hfield",
                {
                    "name": "robodeploy_terrain_hfield",
                    "file": str(terrain.heightfield_path).replace("\\", "/"),
                    "size": f"{float(terrain.size[0])} {float(terrain.size[1])} 1 0.1",
                },
            )
            ET.SubElement(worldbody, "geom", {"name": "terrain", "type": "hfield", "hfield": "robodeploy_terrain_hfield"})

    def _attach_lights(self, world: WorldSpec) -> None:
        if not world.lights:
            return
        worldbody = self._child("worldbody")
        for idx, light in enumerate(world.lights):
            attrs = {
                "name": f"light_{idx}",
                "pos": _fmt(light.position),
                "dir": _fmt(light.direction),
                "diffuse": _fmt(light.diffuse),
            }
            if light.kind == "directional":
                attrs["directional"] = "true"
            ET.SubElement(worldbody, "light", attrs)

    def _attach_prop(self, prop: PropConfig) -> None:
        if prop.asset and prop.asset.get(AssetFormat.MJCF):
            ET.SubElement(self._child("worldbody"), "include", {"file": str(prop.asset[AssetFormat.MJCF]).replace("\\", "/")})
            return

        parent = self._find_body(prop.parent_link) if prop.parent_link else self._child("worldbody")
        body = ET.SubElement(
            parent,
            "body",
            {"name": prop.name, "pos": _fmt(prop.position), "quat": _fmt(prop.orientation)},
        )
        if not prop.is_fixed:
            ET.SubElement(body, "freejoint", {"name": f"{prop.name}_freejoint"})
        if prop.inertia_diag is not None:
            ET.SubElement(body, "inertial", {"mass": str(float(prop.mass)), "diaginertia": _fmt(prop.inertia_diag)})

        geom = prop.geom or self._geom_from_asset_path(prop)
        if geom is None:
            return

        attrs = {
            "name": f"{prop.name}_geom",
            "type": geom.kind,
            "rgba": _fmt(prop.material.rgba),
            "friction": _fmt(prop.material.friction),
        }
        if not prop.is_fixed and prop.inertia_diag is None:
            attrs["mass"] = str(float(prop.mass))
        if geom.kind == "mesh":
            mesh_path = geom.mesh_path or prop.asset_path
            if not mesh_path:
                return
            mesh_name = f"{prop.name}_mesh"
            ET.SubElement(self._child("asset"), "mesh", {"name": mesh_name, "file": str(mesh_path).replace("\\", "/")})
            attrs.update({"type": "mesh", "mesh": mesh_name})
        else:
            attrs["size"] = _fmt(geom.size)
        ET.SubElement(body, "geom", attrs)

    def _attach_cameras(self, world: WorldSpec) -> None:
        if not world.cameras:
            return
        for cam in world.cameras:
            parent = self._find_body(cam.parent_link) if cam.parent_link else self._child("worldbody")
            ET.SubElement(
                parent,
                "camera",
                {
                    "name": cam.name,
                    "pos": _fmt(cam.position),
                    "quat": _fmt(cam.orientation),
                    "fovy": str(float(cam.fov_deg)),
                },
            )

    def _attach_mounted_camera(self, sensor, config: dict) -> None:  # noqa: ANN001
        camera_name = str(config.get("camera_name", getattr(sensor, "name", "camera")))
        if self._has_named("camera", camera_name):
            return
        mount = getattr(sensor, "mount", None)
        parent = self._parent_for_mount(mount, fallback_body=f"{camera_name}_mount")
        ET.SubElement(
            parent,
            "camera",
            {
                "name": camera_name,
                "pos": _fmt(getattr(mount, "position", (0.0, 0.0, 0.0))),
                "quat": _fmt(getattr(mount, "orientation", (1.0, 0.0, 0.0, 0.0))),
                "fovy": str(float(config.get("fov_deg", config.get("fovy", 60.0)))),
            },
        )

    def _attach_mounted_ft_sensor(self, sensor, config: dict) -> None:  # noqa: ANN001
        name = str(getattr(sensor, "name", "ft_sensor"))
        site_name = str(config.get("site", f"{name}_site"))
        force_name = str(config.get("force_sensor", f"{name}_force"))
        torque_name = str(config.get("torque_sensor", f"{name}_torque"))
        mount = getattr(sensor, "mount", None)
        parent = self._parent_for_mount(mount, fallback_body=f"{name}_mount")
        if not self._has_named("site", site_name):
            ET.SubElement(
                parent,
                "site",
                {
                    "name": site_name,
                    "pos": _fmt(getattr(mount, "position", (0.0, 0.0, 0.0))),
                    "quat": _fmt(getattr(mount, "orientation", (1.0, 0.0, 0.0, 0.0))),
                    "size": str(float(config.get("site_size", 0.01))),
                },
            )
        sensor_root = self._child("sensor")
        if not self._has_named("force", force_name):
            ET.SubElement(sensor_root, "force", {"name": force_name, "site": site_name})
        if not self._has_named("torque", torque_name):
            ET.SubElement(sensor_root, "torque", {"name": torque_name, "site": site_name})

    def _find_body(self, name: str | None) -> ET.Element:
        if not name:
            return self._child("worldbody")
        for body in self.root.iter("body"):
            if body.attrib.get("name") == name:
                return body
        return self._child("worldbody")

    def _parent_for_mount(self, mount, *, fallback_body: str) -> ET.Element:  # noqa: ANN001
        parent_link = getattr(mount, "parent_link", None)
        if parent_link:
            return self._find_body(str(parent_link))
        worldbody = self._child("worldbody")
        for body in list(worldbody):
            if body.tag == "body" and body.attrib.get("name") == fallback_body:
                return body
        return ET.SubElement(worldbody, "body", {"name": fallback_body, "pos": "0 0 0"})

    def _has_named(self, tag: str, name: str) -> bool:
        return any(elem.tag == tag and elem.attrib.get("name") == name for elem in self.root.iter(tag))

    def _geom_from_asset_path(self, prop: PropConfig) -> GeomSpec | None:
        if not prop.asset_path:
            return None
        suffix = Path(prop.asset_path).suffix.lower()
        if suffix == ".xml":
            ET.SubElement(self._child("worldbody"), "include", {"file": str(prop.asset_path).replace("\\", "/")})
            return None
        return GeomSpec(kind="mesh", size=(), mesh_path=prop.asset_path)
