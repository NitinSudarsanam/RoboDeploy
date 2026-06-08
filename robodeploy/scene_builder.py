"""Fluent API for composing backend-agnostic scenes."""

from __future__ import annotations

from robodeploy.backends.camera_presets import get_camera_preset
from robodeploy.backends.lighting_presets import get_lighting_preset
from robodeploy.core.scene_ir import (
    Pose3D,
    SceneIR,
    UnifiedGeom,
    UnifiedLighting,
    UnifiedPhysics,
    UnifiedPropSpec,
    UnifiedTerrain,
    UnifiedVisual,
    ir_to_scene_spec,
)
from robodeploy.core.scene_validator import SceneValidationError, SceneValidator
from robodeploy.core.types import LightSpec, SceneSpec


class SceneBuilder:
    def __init__(self) -> None:
        self._props: list[UnifiedPropSpec] = []
        self._lighting = UnifiedLighting()
        self._terrain = UnifiedTerrain()
        self._gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
        self._table_height: float = 0.0
        self._cameras: tuple = ()

    def add_box(
        self,
        name: str,
        *,
        size: tuple[float, float, float],
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        mass: float = 0.1,
        fixed: bool = False,
        rgba: tuple[float, float, float, float] | None = None,
        layer: int = 0,
        mask: int = 0xFFFF,
    ) -> SceneBuilder:
        return self._add_primitive(
            name,
            UnifiedGeom(kind="box", size=size),
            pos=pos,
            quat=quat,
            mass=mass,
            fixed=fixed,
            rgba=rgba,
            layer=layer,
            mask=mask,
        )

    def add_sphere(
        self,
        name: str,
        *,
        radius: float,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        mass: float = 0.1,
        fixed: bool = False,
        rgba: tuple[float, float, float, float] | None = None,
    ) -> SceneBuilder:
        return self._add_primitive(
            name,
            UnifiedGeom(kind="sphere", size=(radius,)),
            pos=pos,
            quat=quat,
            mass=mass,
            fixed=fixed,
            rgba=rgba,
        )

    def add_cylinder(
        self,
        name: str,
        *,
        radius: float,
        height: float,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        mass: float = 0.1,
        fixed: bool = False,
        rgba: tuple[float, float, float, float] | None = None,
    ) -> SceneBuilder:
        return self._add_primitive(
            name,
            UnifiedGeom(kind="cylinder", size=(radius, height)),
            pos=pos,
            quat=quat,
            mass=mass,
            fixed=fixed,
            rgba=rgba,
        )

    def add_capsule(
        self,
        name: str,
        *,
        radius: float,
        length: float,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        mass: float = 0.1,
        fixed: bool = False,
        rgba: tuple[float, float, float, float] | None = None,
    ) -> SceneBuilder:
        return self._add_primitive(
            name,
            UnifiedGeom(kind="capsule", size=(radius, length)),
            pos=pos,
            quat=quat,
            mass=mass,
            fixed=fixed,
            rgba=rgba,
        )

    def add_mesh(
        self,
        name: str,
        *,
        asset: str,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        mass: float = 0.1,
        fixed: bool = False,
        convex_decomp: bool = True,
        rgba: tuple[float, float, float, float] | None = None,
    ) -> SceneBuilder:
        return self._add_primitive(
            name,
            UnifiedGeom(kind="mesh", mesh_path=asset, convex_decomp=convex_decomp),
            pos=pos,
            quat=quat,
            mass=mass,
            fixed=fixed,
            rgba=rgba,
        )

    def add_plane(
        self,
        name: str = "ground",
        *,
        size: tuple[float, float] = (4.0, 4.0),
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> SceneBuilder:
        return self._add_primitive(
            name,
            UnifiedGeom(kind="plane", size=size),
            pos=pos,
            mass=0.0,
            fixed=True,
            rgba=(0.85, 0.85, 0.85, 1.0),
        )

    def add_table(
        self,
        name: str = "table",
        *,
        size: tuple[float, float, float] = (1.2, 0.8, 0.03),
        height: float = 0.4,
    ) -> SceneBuilder:
        z = float(height) - float(size[2])
        return self.add_box(
            name,
            size=size,
            pos=(0.0, 0.0, z),
            fixed=True,
            rgba=(0.55, 0.45, 0.35, 1.0),
        )

    def add_target(
        self,
        name: str = "target",
        *,
        pos: tuple[float, float, float],
        radius: float = 0.04,
    ) -> SceneBuilder:
        return self.add_sphere(
            name,
            radius=radius,
            pos=pos,
            fixed=True,
            rgba=(0.0, 0.8, 0.0, 0.7),
        )

    def set_lighting(self, preset: str | UnifiedLighting) -> SceneBuilder:
        if isinstance(preset, UnifiedLighting):
            self._lighting = preset
        else:
            lights = tuple(get_lighting_preset(str(preset)))
            self._lighting = UnifiedLighting(preset=str(preset).lower(), lights=lights)  # type: ignore[arg-type]
        return self

    def set_terrain(
        self,
        kind: str,
        *,
        size: tuple[float, float] = (4.0, 4.0),
        heightfield_path: str | None = None,
        procedural_params: dict | None = None,
    ) -> SceneBuilder:
        self._terrain = UnifiedTerrain(
            kind=kind if kind in ("flat", "heightfield", "procedural") else "flat",  # type: ignore[arg-type]
            size=size,
            heightfield_path=heightfield_path,
            procedural_params=dict(procedural_params) if procedural_params else None,
        )
        return self

    def set_cameras(self, preset: str) -> SceneBuilder:
        self._cameras = tuple(get_camera_preset(str(preset)))
        return self

    def set_gravity(self, gx: float, gy: float, gz: float) -> SceneBuilder:
        self._gravity = (float(gx), float(gy), float(gz))
        return self

    def set_table_height(self, height: float) -> SceneBuilder:
        self._table_height = float(height)
        return self

    def validate(self, backend: str | None = None) -> SceneBuilder:
        report = SceneValidator().validate(self.build_ir(), backend or "mujoco")
        if not report.ok:
            raise SceneValidationError(report)
        return self

    def build_ir(self) -> SceneIR:
        return SceneIR(
            props=tuple(self._props),
            lighting=self._lighting,
            terrain=self._terrain,
            gravity=self._gravity,
            cameras=self._cameras,
        )

    def build_spec(self) -> SceneSpec:
        return ir_to_scene_spec(self.build_ir(), table_height=self._table_height)

    def _add_primitive(
        self,
        name: str,
        geom: UnifiedGeom,
        *,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        mass: float = 0.1,
        fixed: bool = False,
        rgba: tuple[float, float, float, float] | None = None,
        layer: int = 0,
        mask: int = 0xFFFF,
        friction_dist: tuple[float, float] | None = None,
    ) -> SceneBuilder:
        visual = UnifiedVisual(rgba=rgba) if rgba is not None else UnifiedVisual()
        self._props.append(
            UnifiedPropSpec(
                name=name,
                geometry=geom,
                physics=UnifiedPhysics(
                    mass=mass,
                    is_fixed=fixed,
                    collision_layer=layer,
                    collision_mask=mask,
                    friction_dist=friction_dist,
                ),
                visual=visual,
                pose=Pose3D(position=pos, orientation=quat),
            )
        )
        return self
