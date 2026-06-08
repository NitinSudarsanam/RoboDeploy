"""Gazebo SDF scene composition helpers."""

from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from robodeploy.core.procedural_terrain import ProceduralTerrainGenerator
from robodeploy.core.scene_ir import SceneIR, ir_to_world_spec
from robodeploy.core.types import CameraSpec, GeomSpec, LightSpec, PropConfig, WorldSpec


def _fmt(values) -> str:  # noqa: ANN001
    return " ".join(str(float(v)) for v in values)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _quat_to_rpy(quat: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(_clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def _pose(position: tuple[float, float, float], orientation: tuple[float, float, float, float]) -> str:
    return _fmt((*position, *_quat_to_rpy(orientation)))


def _uri(path: str) -> str:
    raw = str(path or "").strip()
    if "://" in raw:
        return raw
    return f"file://{Path(raw).as_posix()}"


class GazeboSceneBuilder:
    """Build an SDF world from RoboDeploy `WorldSpec`."""

    def __init__(self, *, world_name: str = "robodeploy_world") -> None:
        self.world_name = str(world_name)

    @classmethod
    def from_ir(cls, ir: SceneIR, *, world_name: str = "robodeploy_world") -> str:
        """Build SDF XML from unified SceneIR."""
        return cls(world_name=world_name).build(ir_to_world_spec(ir))

    def build(self, world: WorldSpec) -> str:
        world = self._resolve_procedural_terrain(world)
        root = ET.Element("sdf", {"version": "1.9"})
        world_elem = ET.SubElement(root, "world", {"name": self.world_name})
        ET.SubElement(world_elem, "gravity").text = _fmt(world.gravity)
        self._attach_terrain(world_elem, world)
        if world.lights:
            for idx, light in enumerate(world.lights):
                self._attach_light(world_elem, idx, light)
        else:
            self._attach_light(world_elem, 0, LightSpec())
        for prop in world.props:
            self._attach_prop(world_elem, prop)
        for camera in world.cameras:
            self._attach_camera(world_elem, camera)
        return ET.tostring(root, encoding="unicode")

    def write_temp_world(self, world: WorldSpec) -> Path:
        fd, path = tempfile.mkstemp(prefix="robodeploy_gazebo_world_", suffix=".sdf")
        Path(path).write_text(self.build(world), encoding="utf-8")
        try:
            import os

            os.close(fd)
        except Exception:
            pass
        return Path(path)

    def _attach_terrain(self, world_elem: ET.Element, world: WorldSpec) -> None:
        terrain = world.terrain
        model = ET.SubElement(world_elem, "model", {"name": "ground"})
        ET.SubElement(model, "static").text = "true"
        link = ET.SubElement(model, "link", {"name": "ground_link"})
        if terrain.kind == "heightfield" and terrain.heightfield_path:
            geom = ET.SubElement(ET.SubElement(link, "collision", {"name": "ground_collision"}), "geometry")
            heightmap = ET.SubElement(geom, "heightmap")
            ET.SubElement(heightmap, "uri").text = _uri(terrain.heightfield_path)
            ET.SubElement(heightmap, "size").text = _fmt((terrain.size[0], terrain.size[1], 1.0))

            vis_geom = ET.SubElement(ET.SubElement(link, "visual", {"name": "ground_visual"}), "geometry")
            vis_heightmap = ET.SubElement(vis_geom, "heightmap")
            ET.SubElement(vis_heightmap, "uri").text = _uri(terrain.heightfield_path)
            ET.SubElement(vis_heightmap, "size").text = _fmt((terrain.size[0], terrain.size[1], 1.0))
            return

        for tag in ("collision", "visual"):
            geom = ET.SubElement(ET.SubElement(link, tag, {"name": f"ground_{tag}"}), "geometry")
            plane = ET.SubElement(geom, "plane")
            ET.SubElement(plane, "normal").text = "0 0 1"
            ET.SubElement(plane, "size").text = _fmt(terrain.size)

    def _attach_light(self, world_elem: ET.Element, idx: int, light: LightSpec) -> None:
        light_elem = ET.SubElement(
            world_elem,
            "light",
            {"name": f"light_{idx}", "type": "directional" if light.kind == "directional" else light.kind},
        )
        ET.SubElement(light_elem, "pose").text = _fmt((*light.position, 0.0, 0.0, 0.0))
        ET.SubElement(light_elem, "direction").text = _fmt(light.direction)
        diffuse = ET.SubElement(light_elem, "diffuse")
        diffuse.text = _fmt((*light.diffuse, 1.0))

    def _resolve_procedural_terrain(self, world: WorldSpec) -> WorldSpec:
        terrain = world.terrain
        if terrain.kind != "procedural":
            return world
        params = dict(terrain.procedural_params or {})
        png_path = ProceduralTerrainGenerator.to_temp_heightmap(
            size_m=tuple(terrain.size),
            resolution=int(params.get("resolution", 64)),
            seed=int(params.get("seed", 0)),
            max_height_m=float(params.get("max_height_m", 0.25)),
            generator=str(params.get("generator", "perlin")),
            ridges=int(params.get("ridges", 5)),
            num_steps=int(params.get("num_steps", 8)),
        )
        from dataclasses import replace

        new_terrain = replace(
            terrain,
            kind="heightfield",
            heightfield_path=str(png_path),
        )
        return replace(world, terrain=new_terrain)

    def _attach_prop(self, world_elem: ET.Element, prop: PropConfig) -> None:
        geom = prop.geom or self._geom_from_prop(prop)
        if geom is not None and geom.kind == "capsule":
            self._attach_capsule_compound(world_elem, prop, geom)
            return

        model = ET.SubElement(world_elem, "model", {"name": prop.name})
        ET.SubElement(model, "static").text = "true" if prop.is_fixed else "false"
        ET.SubElement(model, "pose").text = _pose(prop.position, prop.orientation)
        link = ET.SubElement(model, "link", {"name": f"{prop.name}_link"})

        if geom is None:
            return

        if not prop.is_fixed:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass").text = str(float(prop.mass))

        for tag in ("collision", "visual"):
            elem = ET.SubElement(link, tag, {"name": f"{prop.name}_{tag}"})
            geom_elem = ET.SubElement(elem, "geometry")
            self._attach_geom(geom_elem, geom, prop.asset_path)
            if tag == "visual":
                material = ET.SubElement(elem, "material")
                ET.SubElement(material, "ambient").text = _fmt(prop.material.rgba)
                ET.SubElement(material, "diffuse").text = _fmt(prop.material.rgba)

    def _attach_camera(self, world_elem: ET.Element, camera: CameraSpec) -> None:
        model = ET.SubElement(world_elem, "model", {"name": camera.name})
        ET.SubElement(model, "static").text = "true"
        ET.SubElement(model, "pose").text = _pose(camera.position, camera.orientation)
        link = ET.SubElement(model, "link", {"name": f"{camera.name}_link"})
        sensor = ET.SubElement(link, "sensor", {"name": camera.name, "type": "camera"})
        ET.SubElement(sensor, "always_on").text = "true"
        ET.SubElement(sensor, "visualize").text = "false"
        camera_elem = ET.SubElement(sensor, "camera")
        ET.SubElement(camera_elem, "horizontal_fov").text = str(math.radians(float(camera.fov_deg)))
        image = ET.SubElement(camera_elem, "image")
        ET.SubElement(image, "width").text = str(int(camera.resolution[0]))
        ET.SubElement(image, "height").text = str(int(camera.resolution[1]))
        ET.SubElement(image, "format").text = "R8G8B8"
        clip = ET.SubElement(camera_elem, "clip")
        ET.SubElement(clip, "near").text = "0.01"
        ET.SubElement(clip, "far").text = "100.0"

    def _attach_capsule_compound(self, world_elem: ET.Element, prop: PropConfig, geom: GeomSpec) -> None:
        """Approximate SDF capsule with cylinder + two end-cap spheres."""
        radius = float(geom.size[0]) if geom.size else 0.05
        length = float(geom.size[1]) if len(geom.size) > 1 else 0.1
        model = ET.SubElement(world_elem, "model", {"name": prop.name})
        ET.SubElement(model, "static").text = "true" if prop.is_fixed else "false"
        ET.SubElement(model, "pose").text = _pose(prop.position, prop.orientation)
        link = ET.SubElement(model, "link", {"name": f"{prop.name}_link"})
        if not prop.is_fixed:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass").text = str(float(prop.mass))
        parts = (
            ("cyl", f"0 0 0 0 0 0", "cylinder", {"radius": radius, "length": length}),
            ("cap_top", f"0 0 {length / 2.0} 0 0 0", "sphere", {"radius": radius}),
            ("cap_bot", f"0 0 {-length / 2.0} 0 0 0", "sphere", {"radius": radius}),
        )
        for tag in ("collision", "visual"):
            for suffix, pose_txt, prim, attrs in parts:
                elem = ET.SubElement(link, tag, {"name": f"{prop.name}_{suffix}_{tag}"})
                ET.SubElement(elem, "pose").text = pose_txt
                geom_elem = ET.SubElement(elem, "geometry")
                prim_elem = ET.SubElement(geom_elem, prim)
                for key, value in attrs.items():
                    ET.SubElement(prim_elem, key).text = str(float(value))
                if tag == "visual":
                    material = ET.SubElement(elem, "material")
                    ET.SubElement(material, "ambient").text = _fmt(prop.material.rgba)
                    ET.SubElement(material, "diffuse").text = _fmt(prop.material.rgba)

    def _attach_geom(self, geom_elem: ET.Element, geom: GeomSpec, asset_path: str) -> None:
        kind = str(geom.kind)
        if kind == "plane":
            plane = ET.SubElement(geom_elem, "plane")
            size = geom.size if geom.size else (1.0, 1.0)
            ET.SubElement(plane, "normal").text = "0 0 1"
            ET.SubElement(plane, "size").text = _fmt(size[:2])
            return
        if kind == "box":
            box = ET.SubElement(geom_elem, "box")
            ET.SubElement(box, "size").text = _fmt(tuple(float(v) * 2.0 for v in geom.size[:3]))
            return
        if kind == "sphere":
            sphere = ET.SubElement(geom_elem, "sphere")
            ET.SubElement(sphere, "radius").text = str(float(geom.size[0]))
            return
        if kind == "cylinder":
            cylinder = ET.SubElement(geom_elem, "cylinder")
            ET.SubElement(cylinder, "radius").text = str(float(geom.size[0]))
            ET.SubElement(cylinder, "length").text = str(float(geom.size[1]))
            return
        if kind == "capsule":
            capsule = ET.SubElement(geom_elem, "capsule")
            ET.SubElement(capsule, "radius").text = str(float(geom.size[0]))
            ET.SubElement(capsule, "length").text = str(float(geom.size[1]))
            return

        mesh = ET.SubElement(geom_elem, "mesh")
        ET.SubElement(mesh, "uri").text = _uri(geom.mesh_path or asset_path)

    @staticmethod
    def _geom_from_prop(prop: PropConfig) -> GeomSpec | None:
        path = str(prop.asset_path or "")
        if not path:
            return None
        return GeomSpec(kind="mesh", size=(), mesh_path=path)
