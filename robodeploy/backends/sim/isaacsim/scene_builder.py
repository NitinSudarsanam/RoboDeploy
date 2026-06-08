"""Isaac Sim USD scene composition from unified Scene IR."""

from __future__ import annotations

from robodeploy.core.procedural_terrain import ProceduralTerrainGenerator
from robodeploy.core.scene_ir import SceneIR, ir_to_world
from robodeploy.core.types import PropConfig, WorldSpec


class IsaacSceneBuilder:
    """Load Scene IR / WorldSpec props into an active USD stage."""

    def __init__(self, *, warnings: list[str] | None = None) -> None:
        self._warnings = warnings if warnings is not None else []

    @staticmethod
    def planned_prop_geom_counts(ir: SceneIR) -> dict[str, int]:
        """Offline geom plan: one USD prim per prop (matches MuJoCo logical count)."""
        from robodeploy.core.scene_ir import logical_geom_count

        return {prop.name: logical_geom_count(prop) for prop in ir.props}

    def from_ir(self, ir: SceneIR, *, props_scope: str = "/World/RoboDeployProps") -> dict[str, str]:
        return self.load_world(ir_to_world(ir), props_scope=props_scope)

    def load_world(self, world: WorldSpec, *, props_scope: str = "/World/RoboDeployProps") -> dict[str, str]:
        world = self._resolve_procedural_terrain(world)
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import Gf, UsdGeom, UsdLux  # type: ignore[import-not-found]
        except Exception as exc:
            if world.props or world.cameras or world.lights:
                self._warnings.append(f"Isaac USD scene loading unavailable: {exc}")
            return {}

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._warnings.append("Isaac USD scene loading skipped: no active stage.")
            return {}

        prop_paths: dict[str, str] = {}
        UsdGeom.Xform.Define(stage, props_scope)
        for prop in world.props:
            path = f"{props_scope}/{prop.name}"
            try:
                prim_api = self._define_prop_prim(stage, UsdGeom, Gf, prop, path)
                if prim_api is not None:
                    self._set_usd_pose(UsdGeom.XformCommonAPI(prim_api), Gf, prop.position, prop.orientation)
                    prop_paths[prop.name] = path
            except Exception as exc:
                self._warnings.append(f"Isaac prop '{prop.name}' was not loaded into USD: {exc}")

        for idx, light in enumerate(world.lights):
            try:
                path = f"/World/RoboDeployLight_{idx}"
                if light.kind == "directional":
                    light_api = UsdLux.DistantLight.Define(stage, path)
                else:
                    light_api = UsdLux.SphereLight.Define(stage, path)
                UsdGeom.XformCommonAPI(light_api).SetTranslate(Gf.Vec3d(*[float(v) for v in light.position]))
            except Exception as exc:
                self._warnings.append(f"Isaac light {idx} was not loaded into USD: {exc}")

        for cam in world.cameras:
            try:
                cam_api = UsdGeom.Camera.Define(stage, f"/World/{cam.name}")
                self._set_usd_pose(UsdGeom.XformCommonAPI(cam_api), Gf, cam.position, cam.orientation)
                cam_api.GetVerticalApertureAttr().Set(float(cam.fov_deg))
            except Exception as exc:
                self._warnings.append(f"Isaac camera '{cam.name}' was not loaded into USD: {exc}")

        if world.terrain.kind == "heightfield" and world.terrain.heightfield_path:
            try:
                self._attach_heightfield_mesh(stage, UsdGeom, world.terrain, path="/World/RoboDeployTerrain")
            except Exception as exc:
                self._warnings.append(f"Isaac heightfield terrain mesh failed: {exc}")

        return prop_paths

    def _attach_heightfield_mesh(self, stage, UsdGeom, terrain, *, path: str) -> None:  # noqa: ANN001,N803
        """Build a collision/visual mesh from a grayscale heightmap PNG."""
        import numpy as np

        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for Isaac heightfield terrain meshes.") from exc

        img = Image.open(str(terrain.heightfield_path))
        heights = np.asarray(img, dtype=np.float32)
        if heights.ndim == 3:
            heights = heights[..., 0]
        heights = heights / max(float(heights.max()), 1.0)
        max_height_m = float(getattr(terrain, "max_height_m", 0.25) or 0.25)
        res_y, res_x = heights.shape
        size_x, size_y = float(terrain.size[0]), float(terrain.size[1])

        points: list[tuple[float, float, float]] = []
        for iy in range(res_y):
            for ix in range(res_x):
                x = (ix / max(res_x - 1, 1) - 0.5) * size_x
                y = (iy / max(res_y - 1, 1) - 0.5) * size_y
                z = float(heights[iy, ix]) * max_height_m
                points.append((x, y, z))

        face_vertex_counts: list[int] = []
        face_vertex_indices: list[int] = []
        for iy in range(res_y - 1):
            for ix in range(res_x - 1):
                i0 = iy * res_x + ix
                i1 = i0 + 1
                i2 = i0 + res_x
                i3 = i2 + 1
                face_vertex_counts.extend((3, 3))
                face_vertex_indices.extend((i0, i1, i2, i1, i3, i2))

        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
        try:
            from pxr import PhysxSchema  # type: ignore[import-not-found]

            prim = mesh.GetPrim()
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
        except Exception:
            pass

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

        return replace(
            world,
            terrain=replace(terrain, kind="heightfield", heightfield_path=str(png_path)),
        )

    def _define_prop_prim(self, stage, UsdGeom, Gf, prop: PropConfig, path: str):  # noqa: ANN001,N803
        geom = prop.geom
        if geom is not None and geom.kind == "sphere":
            prim_api = UsdGeom.Sphere.Define(stage, path)
            if geom.size:
                prim_api.GetRadiusAttr().Set(float(geom.size[0]))
            return prim_api
        if geom is not None and geom.kind == "cylinder":
            prim_api = UsdGeom.Cylinder.Define(stage, path)
            if geom.size:
                prim_api.GetRadiusAttr().Set(float(geom.size[0]))
            if len(geom.size) > 1:
                prim_api.GetHeightAttr().Set(float(geom.size[1]) * 2.0)
            return prim_api
        if geom is not None and geom.kind == "capsule":
            prim_api = UsdGeom.Capsule.Define(stage, path)
            if geom.size:
                prim_api.GetRadiusAttr().Set(float(geom.size[0]))
            if len(geom.size) > 1:
                prim_api.GetHeightAttr().Set(float(geom.size[1]))
            return prim_api
        if geom is not None and geom.kind == "plane":
            prim_api = UsdGeom.Plane.Define(stage, path)
            size = geom.size if geom.size else (1.0, 1.0)
            prim_api.GetWidthAttr().Set(float(size[0]))
            prim_api.GetLengthAttr().Set(float(size[1] if len(size) > 1 else size[0]))
            return prim_api
        if geom is not None and geom.kind == "mesh" and geom.mesh_path:
            prim = stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddReference(str(geom.mesh_path).replace("\\", "/"))
            return UsdGeom.Xformable(prim)
        prim_api = UsdGeom.Cube.Define(stage, path)
        size = tuple(getattr(geom, "size", ()) or (0.05, 0.05, 0.05))
        UsdGeom.XformCommonAPI(prim_api).SetScale(Gf.Vec3d(*(float(v) * 2.0 for v in size[:3])))
        return prim_api

    @staticmethod
    def _set_usd_pose(xform_api, Gf, position, orientation=None) -> None:  # noqa: ANN001,N803
        import math

        xform_api.SetTranslate(Gf.Vec3d(*[float(v) for v in position]))
        if orientation is not None and hasattr(xform_api, "SetRotate"):
            w, x, y, z = (float(v) for v in orientation)
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            sinp = 2.0 * (w * y - z * x)
            pitch = math.asin(max(-1.0, min(1.0, sinp)))
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            xform_api.SetRotate((math.degrees(roll), math.degrees(pitch), math.degrees(yaw)))
