"""Unified Scene IR — backend-agnostic scene representation (Goal 1 / Goal 6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from robodeploy.core.spaces import AssetFormat
from robodeploy.core.types import CameraSpec, GeomSpec, LightSpec, MaterialSpec, PropConfig, TerrainSpec, WorldSpec

GeomKind = Literal["box", "sphere", "cylinder", "capsule", "mesh", "plane", "heightfield"]


@dataclass(frozen=True)
class UnifiedGeom:
    kind: GeomKind
    size: tuple[float, ...] = ()
    mesh_path: str | None = None
    heightfield_path: str | None = None
    convex_decomp: bool = False


@dataclass(frozen=True)
class UnifiedPhysics:
    mass: float = 0.1
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    restitution: float = 0.0
    damping: float = 0.0
    collision_layer: int = 0
    collision_mask: int = 0xFFFF
    friction_dist: tuple[float, float] | None = None
    is_fixed: bool = False


@dataclass(frozen=True)
class UnifiedVisual:
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    material: str | None = None
    texture_path: str | None = None


@dataclass(frozen=True)
class Pose3D:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class UnifiedPropSpec:
    name: str
    geometry: UnifiedGeom
    physics: UnifiedPhysics = field(default_factory=UnifiedPhysics)
    visual: UnifiedVisual = field(default_factory=UnifiedVisual)
    pose: Pose3D = field(default_factory=Pose3D)
    variants: dict[str, str] = field(default_factory=dict)
    parent_frame: str | None = None


@dataclass(frozen=True)
class UnifiedLighting:
    preset: Literal["minimal", "bright", "dark", "randomized"] | None = None
    lights: tuple[LightSpec, ...] = ()


@dataclass(frozen=True)
class UnifiedTerrain:
    kind: Literal["flat", "heightfield", "procedural"] = "flat"
    size: tuple[float, float] = (4.0, 4.0)
    heightfield_path: str | None = None
    procedural_params: dict | None = None


@dataclass(frozen=True)
class SceneIR:
    props: tuple[UnifiedPropSpec, ...]
    lighting: UnifiedLighting = field(default_factory=UnifiedLighting)
    terrain: UnifiedTerrain = field(default_factory=UnifiedTerrain)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    cameras: tuple[CameraSpec, ...] = ()


def geom_from_prop(prop: PropConfig) -> UnifiedGeom:
    if prop.geom is not None:
        kind = str(prop.geom.kind)
        if kind not in ("box", "sphere", "cylinder", "capsule", "mesh", "plane", "heightfield"):
            kind = "box"
        return UnifiedGeom(
            kind=kind,  # type: ignore[arg-type]
            size=tuple(prop.geom.size),
            mesh_path=prop.geom.mesh_path,
        )
    if prop.asset_path:
        return UnifiedGeom(kind="mesh", mesh_path=prop.asset_path)
    return UnifiedGeom(kind="box", size=(0.05, 0.05, 0.05))


def prop_to_ir(prop: PropConfig) -> UnifiedPropSpec:
    variants: dict[str, str] = {}
    if prop.asset:
        for fmt, path in prop.asset.items():
            key = fmt.value if isinstance(fmt, AssetFormat) else str(fmt)
            variants[key] = str(path)
    if prop.asset_path and "mesh" not in variants:
        variants["mesh"] = str(prop.asset_path)
    return UnifiedPropSpec(
        name=prop.name,
        geometry=geom_from_prop(prop),
        physics=UnifiedPhysics(
            mass=float(prop.mass),
            friction=tuple(prop.material.friction),
            collision_layer=int(prop.collision_layer),
            collision_mask=int(prop.collision_mask),
            friction_dist=tuple(prop.friction_dist) if prop.friction_dist is not None else None,
            is_fixed=bool(prop.is_fixed),
        ),
        visual=UnifiedVisual(rgba=tuple(prop.material.rgba), texture_path=prop.material.texture),
        pose=Pose3D(position=tuple(prop.position), orientation=tuple(prop.orientation)),
        variants=variants,
        parent_frame=prop.parent_link,
    )


def world_to_ir(world: WorldSpec, *, lighting_preset: str | None = None) -> SceneIR:
    return SceneIR(
        props=tuple(prop_to_ir(p) for p in world.props),
        lighting=UnifiedLighting(
            preset=lighting_preset if lighting_preset in ("minimal", "bright", "dark", "randomized") else None,
            lights=tuple(world.lights),
        ),
        terrain=UnifiedTerrain(
            kind=world.terrain.kind,
            size=tuple(world.terrain.size),
            heightfield_path=world.terrain.heightfield_path,
            procedural_params=world.terrain.procedural_params,
        ),
        gravity=tuple(world.gravity),
        cameras=tuple(world.cameras),
    )


def ir_to_prop(prop: UnifiedPropSpec) -> PropConfig:
    geom_kind = prop.geometry.kind
    if geom_kind == "heightfield":
        geom_kind = "box"
    asset: dict[AssetFormat, str] = {}
    for key, path in prop.variants.items():
        try:
            asset[AssetFormat(key)] = path
        except ValueError:
            continue
    return PropConfig(
        name=prop.name,
        asset_path=prop.geometry.mesh_path or prop.variants.get("mesh", ""),
        position=prop.pose.position,
        orientation=prop.pose.orientation,
        mass=prop.physics.mass,
        is_fixed=prop.physics.is_fixed,
        geom=GeomSpec(
            kind=geom_kind,  # type: ignore[arg-type]
            size=tuple(prop.geometry.size),
            mesh_path=prop.geometry.mesh_path,
        ),
        material=MaterialSpec(rgba=prop.visual.rgba, friction=prop.physics.friction, texture=prop.visual.texture_path),
        asset=asset or None,
        parent_link=prop.parent_frame,
        collision_layer=int(prop.physics.collision_layer),
        collision_mask=int(prop.physics.collision_mask),
        friction_dist=prop.physics.friction_dist,
    )


def ir_to_world(ir: SceneIR) -> WorldSpec:
    terrain = TerrainSpec(
        kind=ir.terrain.kind,
        size=ir.terrain.size,
        heightfield_path=ir.terrain.heightfield_path,
        procedural_params=dict(ir.terrain.procedural_params or {}) if ir.terrain.procedural_params else None,
    )
    return WorldSpec(
        props=[ir_to_prop(p) for p in ir.props],
        lights=list(ir.lighting.lights),
        cameras=list(ir.cameras),
        terrain=terrain,
        gravity=ir.gravity,
    )


ir_to_world_spec = ir_to_world


def scene_spec_to_ir(spec) -> SceneIR:
    """Convert ``SceneSpec`` to unified IR."""
    preset = spec.lighting if spec.lighting in ("minimal", "bright", "dark", "randomized") else None
    return world_to_ir(spec.to_world(), lighting_preset=preset)


def logical_geom_count(prop: UnifiedPropSpec) -> int:
    """One logical collision primitive per prop (capsule stays 1 at IR level)."""
    return 1


def backend_collision_geom_count(prop: UnifiedPropSpec, *, backend: str) -> int:
    """Collision primitive count after backend-specific decomposition."""
    if backend.lower() in ("gazebo", "ros2") and prop.geometry.kind == "capsule":
        return 3
    return logical_geom_count(prop)


def ir_prop_count(ir: SceneIR) -> int:
    return len(ir.props)


def ir_logical_geom_total(ir: SceneIR) -> int:
    return sum(logical_geom_count(p) for p in ir.props)


def ir_backend_geom_total(ir: SceneIR, *, backend: str) -> int:
    return sum(backend_collision_geom_count(p, backend=backend) for p in ir.props)


def count_mujoco_prop_bodies(xml: str, prop_names: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in prop_names}
    for name in prop_names:
        token = f'body name="{name}"'
        counts[name] = xml.count(token)
    return counts


def count_gazebo_collision_geoms(sdf: str, prop_names: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in prop_names}
    for name in prop_names:
        marker = f'<model name="{name}"'
        start = sdf.find(marker)
        if start < 0:
            continue
        end = sdf.find("<model", start + len(marker))
        chunk = sdf[start:end] if end >= 0 else sdf[start:]
        counts[name] = chunk.count("<collision")
    return counts


def ir_prop_positions(ir: SceneIR) -> dict[str, tuple[float, float, float]]:
    """Ground-truth prop positions from unified IR."""
    return {prop.name: tuple(float(v) for v in prop.pose.position) for prop in ir.props}


def extract_mujoco_prop_positions(xml: str, prop_names: list[str]) -> dict[str, tuple[float, float, float]]:
    """Parse MJCF body ``pos`` attributes for named props."""
    import re

    positions: dict[str, tuple[float, float, float]] = {}
    for name in prop_names:
        match = re.search(rf'<body\s+name="{re.escape(name)}"[^>]*\s+pos="([^"]+)"', xml)
        if not match:
            continue
        parts = [float(v) for v in match.group(1).split()]
        if len(parts) >= 3:
            positions[name] = (parts[0], parts[1], parts[2])
    return positions


def extract_gazebo_prop_positions(sdf: str, prop_names: list[str]) -> dict[str, tuple[float, float, float]]:
    """Parse SDF model ``pose`` xyz (roll-pitch-yaw suffix ignored)."""
    import re

    positions: dict[str, tuple[float, float, float]] = {}
    for name in prop_names:
        marker = f'<model name="{name}"'
        start = sdf.find(marker)
        if start < 0:
            continue
        end = sdf.find("<model", start + len(marker))
        chunk = sdf[start:end] if end >= 0 else sdf[start:]
        match = re.search(r"<pose>([^<]+)</pose>", chunk)
        if not match:
            continue
        parts = [float(v) for v in match.group(1).split()]
        if len(parts) >= 3:
            positions[name] = (parts[0], parts[1], parts[2])
    return positions


def assert_cross_backend_pose_equivalence(
    ir: SceneIR,
    *,
    mjcf: str | None = None,
    sdf: str | None = None,
    atol: float = 1e-3,
) -> None:
    """Raise ``AssertionError`` when backend-emitted poses diverge from IR (>1 mm default)."""
    import numpy as np

    expected = ir_prop_positions(ir)
    prop_names = list(expected.keys())
    if mjcf is not None:
        actual = extract_mujoco_prop_positions(mjcf, prop_names)
        for name, pos in expected.items():
            if name not in actual:
                raise AssertionError(f"MuJoCo missing prop position for '{name}'")
            np.testing.assert_allclose(actual[name], pos, atol=atol, err_msg=f"MuJoCo pose mismatch for '{name}'")
    if sdf is not None:
        actual = extract_gazebo_prop_positions(sdf, prop_names)
        for name, pos in expected.items():
            if name not in actual:
                raise AssertionError(f"Gazebo missing prop position for '{name}'")
            np.testing.assert_allclose(actual[name], pos, atol=atol, err_msg=f"Gazebo pose mismatch for '{name}'")


def ir_to_scene_spec(ir: SceneIR, *, table_height: float = 0.0, lighting: str | None = None):
    """Convert IR back to ``SceneSpec`` for legacy backends."""
    from robodeploy.backends.lighting_presets import get_lighting_preset
    from robodeploy.core.types import SceneSpec

    world = ir_to_world(ir)
    if not world.lights:
        if ir.lighting.lights:
            world.lights = list(ir.lighting.lights)
        elif ir.lighting.preset:
            world.lights = get_lighting_preset(ir.lighting.preset)
    lighting_name = lighting or ir.lighting.preset or "default"
    return SceneSpec(
        props=list(world.props),
        table_height=float(table_height),
        lighting=str(lighting_name),
        world=world,
    )
