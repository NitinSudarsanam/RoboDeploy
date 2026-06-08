"""IsaacSimBackend — optional Isaac Sim simulation backend.

This module isolates Isaac imports so the library remains importable on
machines that do not have Isaac Sim installed.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol

from robodeploy.backends.base import BackendBase
from robodeploy.core.registry import register_backend
from robodeploy.core.spaces import ActionSpace, AssetFormat
from robodeploy.core.types import Action, Observation, Pose3D, SceneSpec, WorldSpec

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.core.robot import Robot
    from robodeploy.description.base import RobotDescription


class _SimulationAppLike(Protocol):
    def update(self) -> None: ...
    def close(self, *args: Any, **kwargs: Any) -> None: ...
    def run_coroutine(self, coroutine, run_until_complete: bool = True): ...  # noqa: ANN001


@dataclass(frozen=True)
class _IsaacLaunchConfig:
    headless: bool
    width: int
    height: int
    renderer: Optional[str]
    experience: str


@dataclass
class _IsaacHandles:
    """Late-bound Isaac/Omni handles (keeps imports out of module import path)."""

    SimulationApp: type
    World: type
    SingleArticulation: type
    ArticulationAction: type
    omni_kit_commands: Any


@dataclass
class _IsaacRobotRuntime:
    robot_id: str
    description: "RobotDescription"
    articulation: Any
    prim_path: str
    sensors: list


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _quat_to_xyz_degrees(quat: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(_clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


@register_backend("isaacsim")
class IsaacSimBackend(BackendBase):
    """Isaac Sim backend (optional; runtime-guarded)."""

    is_real = False
    sensor_backend_name = "isaacsim"
    control_hz = 100.0
    supported_action_spaces = [ActionSpace.JOINT_POS]

    def initialize_multi(self, robots, scene: SceneSpec, shared_sensors) -> None:  # type: ignore[override]
        if not robots:
            raise ValueError("IsaacSimBackend.initialize_multi requires at least one robot.")
        if len(robots) == 1:
            robot = robots[0]
            self._robot_id = str(robot.robot_id or "robot0")
            self._multi_mode = False
            super().initialize(robot.description, scene, [*robot.sensors, *shared_sensors])
            return

        self._multi_mode = True
        self._description = robots[0].description
        self._scene = scene
        sensors: list[ISensor] = []
        seen: set[int] = set()
        for robot in robots:
            for sensor in robot.sensors:
                if id(sensor) not in seen:
                    sensors.append(sensor)
                    seen.add(id(sensor))
        for sensor in shared_sensors or []:
            if id(sensor) not in seen:
                sensors.append(sensor)
                seen.add(id(sensor))
        self._sensors = sensors
        self._asset_selections.clear()
        self._load_multi(robots, scene, sensors, list(shared_sensors or []))
        self._initialized = True

    def reset_multi(self, robot_ids: list[str] | None = None) -> list[Observation]:
        if not getattr(self, "_multi_mode", False):
            del robot_ids
            return [self.reset()]
        self._require_initialized("reset_multi")
        headless = bool(self.config.get("headless", True))
        self._ensure_physics_ready(self._world, self._simulation_app)
        ids = robot_ids or list(self._robot_order)
        for rid in ids:
            runtime = self._robot_runtimes[rid]
            try:
                runtime.articulation.initialize()
            except Exception as exc:
                raise RuntimeError(f"Failed to reinitialize Isaac articulation for '{rid}'.") from exc
            home = getattr(runtime.description, "home_qpos", None)
            if home is not None:
                import numpy as np

                q = np.asarray(home, dtype=np.float32)
                runtime.articulation.apply_action(self._isaac.ArticulationAction(joint_positions=q))
        for _ in range(5 * self._steps_per_control):
            self._world.step(render=not headless)
            self._simulation_app.update()
        self._sim_time = 0.0
        self._episode_count += 1
        self._step_count = 0
        out: list[Observation] = []
        for rid in self._robot_order:
            obs = self._obs_for_robot(rid)
            out.append(obs)
            if getattr(self, "_rviz", None) is not None:
                self._rviz.publish_robot_state(rid, obs)
        if getattr(self, "_rviz", None) is not None:
            self._rviz.reset()
            if self._latest_viz_payload is not None:
                self._rviz.publish_task_viz(self._latest_viz_payload)
        return out

    def step_multi(self, actions: list[Action]) -> list[Observation]:
        if not getattr(self, "_multi_mode", False):
            if len(actions) != 1:
                raise ValueError(f"IsaacSimBackend.step_multi expected 1 action, got {len(actions)}.")
            return [self.step(actions[0])]
        self._require_initialized("step_multi")
        if len(actions) != len(self._robot_order):
            raise ValueError(
                f"IsaacSimBackend.step_multi expected {len(self._robot_order)} actions, got {len(actions)}."
            )
        headless = bool(self.config.get("headless", True))
        for rid, action in zip(self._robot_order, actions):
            if action.joint_positions is None:
                continue
            import numpy as np

            q = np.asarray(action.joint_positions, dtype=np.float32)
            self._robot_runtimes[rid].articulation.apply_action(self._isaac.ArticulationAction(joint_positions=q))
        for _ in range(self._steps_per_control):
            self._world.step(render=not headless)
            self._simulation_app.update()
            self._sim_time += self._physics_dt
        out: list[Observation] = []
        for rid in self._robot_order:
            obs = self._obs_for_robot(rid)
            out.append(obs)
            if getattr(self, "_rviz", None) is not None:
                self._rviz.publish_robot_state(rid, obs)
        if getattr(self, "_rviz", None) is not None and self._latest_viz_payload is not None:
            self._rviz.publish_task_viz(self._latest_viz_payload)
        self._step_count += 1
        return out

    def get_obs_multi(self) -> list[Observation]:
        if not getattr(self, "_multi_mode", False):
            return [self.get_obs()]
        self._require_initialized("get_obs_multi")
        return [self._obs_for_robot(rid) for rid in self._robot_order]

    def _load(self, description: RobotDescription, scene: SceneSpec, sensors: list[ISensor]) -> None:
        self._multi_mode = False
        self._sensors = list(sensors)
        self._warnings: list[str] = []
        world = scene.to_world()
        self._world_spec = world
        self._scene_prop_poses = {
            prop.name: (tuple(prop.position), tuple(prop.orientation))
            for prop in world.props
        }
        self._scene_prop_paths: dict[str, str] = {}
        launch = self._parse_launch_config()
        self._simulation_app, self._isaac = self._launch_kit(launch)
        self._world = self._create_world(self._isaac)
        from robodeploy.backends.sim.isaacsim.scene_builder import IsaacSceneBuilder

        builder = IsaacSceneBuilder(warnings=self._warnings)
        self._scene_prop_paths = builder.load_world(world)

        # Prefer USD when available (USD-first), but keep a safe URDF fallback.
        usd_prefer = bool(self.config.get("usd_prefer", True))
        usd_fallback = bool(self.config.get("usd_fallback_to_urdf", True))
        robot_loaded = False
        if usd_prefer:
            try:
                usd_path = self._resolve_asset_path(self._robot_id, description, AssetFormat.USD, variant="sim")
                if usd_path.exists():
                    suffix = usd_path.suffix.lower()
                    if suffix in (".usd", ".usda", ".usdc"):
                        self._robot_prim_path = self._import_usd_robot(usd_path)
                        robot_loaded = True
                    elif not usd_fallback:
                        raise NotImplementedError(f"Unsupported USD asset extension for {usd_path}.")
                    else:
                        self._warnings.append(f"USD asset present ({usd_path}), falling back to URDF import.")
            except FileNotFoundError:
                pass

        if not robot_loaded:
            urdf_path = self._resolve_asset_path(self._robot_id, description, AssetFormat.URDF, variant="sim")
            self._robot_prim_path = self._import_urdf_robot(self._isaac, urdf_path)
        self._robot = self._isaac.SingleArticulation(prim_path=self._robot_prim_path, name=self._robot_id)

        self._ensure_physics_ready(self._world, self._simulation_app)
        self._initialize_articulation(self._robot)

        self._init_timing()
        self._seed_home_pose(description, headless=launch.headless)

        # Optional RViz sidecar (publishes RoboDeploy-standard topics).
        self._rviz = None
        self._latest_viz_payload: Optional[dict] = None
        rviz_cfg = (self.config.get("rviz") or {}) if isinstance(self.config.get("rviz"), dict) else {}
        if bool(rviz_cfg.get("enabled", False)):
            from robodeploy.viz.rviz_publisher import RvizPublisher

            self._rviz = RvizPublisher(
                fixed_frame=str(rviz_cfg.get("fixed_frame", "world")),
                publish_hz=float(rviz_cfg.get("publish_hz", 10.0)),
                namespace="/robodeploy",
                base_frame=description.ros_base_frame_id(),
            )
            self._rviz.start()
            try:
                self._rviz.publish_scene(scene)
            except Exception:
                self._warnings.append("RViz scene publish failed (non-fatal).")

    def _load_world_spec_into_stage(self, world) -> dict[str, str]:  # noqa: ANN001
        """Load props/cameras/lights from WorldSpec into the active USD stage."""
        from robodeploy.backends.sim.isaacsim.scene_builder import IsaacSceneBuilder

        builder = IsaacSceneBuilder(warnings=getattr(self, "_warnings", None))
        self._scene_prop_paths = builder.load_world(world)
        return self._scene_prop_paths

    def _load_multi(
        self,
        robots: list["Robot"],
        scene: SceneSpec,
        sensors: list[ISensor],
        shared_sensors: list[ISensor],
    ) -> None:
        del sensors
        self._warnings = []
        world = scene.to_world()
        self._world_spec = world
        self._scene_prop_poses = {
            prop.name: (tuple(prop.position), tuple(prop.orientation))
            for prop in world.props
        }
        self._scene_prop_paths = {}
        self._robot_runtimes: dict[str, _IsaacRobotRuntime] = {}
        self._robot_order = [str(robot.robot_id or f"robot{i}") for i, robot in enumerate(robots)]
        launch = self._parse_launch_config()
        self._simulation_app, self._isaac = self._launch_kit(launch)
        self._world = self._create_world(self._isaac)
        self._load_world_spec_into_stage(world)
        usd_prefer = bool(self.config.get("usd_prefer", True))
        usd_fallback = bool(self.config.get("usd_fallback_to_urdf", True))

        for robot in robots:
            robot_id = str(robot.robot_id or self._robot_order[0])
            prim_path = self._import_robot_asset(
                robot_id,
                robot.description,
                usd_prefer=usd_prefer,
                usd_fallback=usd_fallback,
            )
            articulation = self._isaac.SingleArticulation(prim_path=prim_path, name=robot_id)
            self._robot_runtimes[robot_id] = _IsaacRobotRuntime(
                robot_id=robot_id,
                description=robot.description,
                articulation=articulation,
                prim_path=prim_path,
                sensors=[*robot.sensors, *shared_sensors],
            )

        self._ensure_physics_ready(self._world, self._simulation_app)
        for runtime in self._robot_runtimes.values():
            self._initialize_articulation(runtime.articulation)

        self._init_timing()
        self._robot_id = self._robot_order[0]
        self._robot = self._robot_runtimes[self._robot_id].articulation
        for runtime in self._robot_runtimes.values():
            self._seed_home_pose_for(runtime, headless=launch.headless)

        self._rviz = None
        self._latest_viz_payload = None
        rviz_cfg = (self.config.get("rviz") or {}) if isinstance(self.config.get("rviz"), dict) else {}
        if bool(rviz_cfg.get("enabled", False)):
            from robodeploy.viz.rviz_publisher import RvizPublisher

            self._rviz = RvizPublisher(
                fixed_frame=str(rviz_cfg.get("fixed_frame", "world")),
                publish_hz=float(rviz_cfg.get("publish_hz", 10.0)),
                namespace="/robodeploy",
                base_frame=robots[0].description.ros_base_frame_id(),
            )
            self._rviz.start()
            try:
                self._rviz.publish_scene(scene)
            except Exception:
                self._warnings.append("RViz scene publish failed (non-fatal).")

    def _import_robot_asset(
        self,
        robot_id: str,
        description: "RobotDescription",
        *,
        usd_prefer: bool,
        usd_fallback: bool,
    ) -> str:
        prim_path = f"/World/{robot_id}"
        if usd_prefer:
            try:
                usd_path = self._resolve_asset_path(robot_id, description, AssetFormat.USD, variant="sim")
                if usd_path.exists():
                    suffix = usd_path.suffix.lower()
                    if suffix in (".usd", ".usda", ".usdc"):
                        return self._import_usd_robot(usd_path, prim_path=prim_path)
                    if not usd_fallback:
                        raise NotImplementedError(f"Unsupported USD asset extension for {usd_path}.")
                    self._warnings.append(f"USD asset present ({usd_path}), falling back to URDF import.")
            except FileNotFoundError:
                pass
        urdf_path = self._resolve_asset_path(robot_id, description, AssetFormat.URDF, variant="sim")
        return self._import_urdf_robot(self._isaac, urdf_path, prim_path=prim_path)

    def _obs_for_robot(self, robot_id: str) -> Observation:
        runtime = self._robot_runtimes[robot_id]
        obs = self._build_obs(robot=runtime.articulation, description=runtime.description)
        return self._merge_sensor_data(obs, runtime.sensors)

    def _seed_home_pose_for(self, runtime: _IsaacRobotRuntime, *, headless: bool) -> None:
        home = getattr(runtime.description, "home_qpos", None)
        if home is None:
            return
        import numpy as np

        q = np.asarray(home, dtype=np.float32)
        runtime.articulation.apply_action(self._isaac.ArticulationAction(joint_positions=q))
        for _ in range(5 * self._steps_per_control):
            self._world.step(render=not headless)
            self._simulation_app.update()

    def get_prop_names(self) -> list[str]:
        return sorted(getattr(self, "_scene_prop_poses", {}))

    def get_prop_pose(self, name: str):
        poses = getattr(self, "_scene_prop_poses", {})
        if name not in poses:
            raise KeyError(f"Unknown IsaacSim prop '{name}'.")
        live = self._read_usd_prop_pose(name)
        return live if live is not None else poses[name]

    def set_prop_pose(self, name: str, position, orientation) -> None:  # noqa: ANN001
        poses = getattr(self, "_scene_prop_poses", {})
        if name not in poses:
            raise KeyError(f"Unknown IsaacSim prop '{name}'.")
        poses[name] = (tuple(float(v) for v in position), tuple(float(v) for v in orientation))
        if not self._write_usd_prop_pose(name, position, orientation):
            self._warnings.append(
                f"Prop pose for '{name}' updated in RoboDeploy metadata; Isaac USD prim editing was unavailable."
            )

    def teleport_object(self, name: str, position: tuple[float, float, float]) -> None:
        _, quat = self.get_prop_pose(name)
        self.set_prop_pose(name, position, quat)

    def set_physics_params(self, **kwargs) -> None:
        if "gravity" in kwargs:
            gravity = kwargs["gravity"]
            try:
                self._world.set_gravity([float(v) for v in gravity])
            except Exception as exc:
                raise NotImplementedError(f"IsaacSimBackend could not set gravity: {exc}") from exc
        friction = kwargs.get("friction")
        restitution = kwargs.get("restitution")
        damping = kwargs.get("damping")
        if friction is not None or restitution is not None:
            try:
                import omni.usd  # type: ignore[import-not-found]
                from pxr import PhysxSchema  # type: ignore[import-not-found]

                stage = omni.usd.get_context().get_stage()
                if stage is not None:
                    for path in getattr(self, "_scene_prop_paths", {}).values():
                        prim = stage.GetPrimAtPath(path)
                        if prim is None or not prim.IsValid():
                            continue
                        material = PhysxSchema.PhysxMaterialAPI.Apply(prim)
                        if friction is not None:
                            value = float(friction[0] if isinstance(friction, (list, tuple)) else friction)
                            material.CreateStaticFrictionAttr(value)
                            material.CreateDynamicFrictionAttr(value)
                        if restitution is not None:
                            material.CreateRestitutionAttr(float(restitution))
            except Exception as exc:
                self._warnings.append(f"Isaac physics material tuning unavailable: {exc}")
        if damping is not None:
            self._apply_joint_damping(float(damping))

    def get_diagnostics(self) -> dict:
        return {
            "backend": "isaacsim",
            "warnings": list(getattr(self, "_warnings", [])),
            "props_loaded": sorted(getattr(self, "_scene_prop_paths", {})),
            **self._sensor_diagnostics(),
        }

    # ------------------------------------------------------------------
    # Launch / world / asset loading (kept modular for future USD path)
    # ------------------------------------------------------------------

    def _parse_launch_config(self) -> _IsaacLaunchConfig:
        headless = bool(self.config.get("headless", True))
        width = int(self.config.get("width", 1280))
        height = int(self.config.get("height", 720))
        renderer = self.config.get("renderer", None)
        experience = str(self.config.get("experience", "isaacsim.exp.base.python.kit"))
        return _IsaacLaunchConfig(
            headless=headless,
            width=width,
            height=height,
            renderer=str(renderer) if renderer is not None else None,
            experience=experience,
        )

    def _launch_kit(self, launch: _IsaacLaunchConfig) -> tuple[_SimulationAppLike, _IsaacHandles]:
        # Isaac Sim requires that Kit is initialised before most omni/isaac imports.
        try:
            from isaacsim.simulation_app import SimulationApp  # type: ignore[import-not-found]
        except Exception as exc:
            raise ImportError(
                "IsaacSimBackend requires Isaac Sim's Python environment.\n"
                "Run this using Isaac's python.bat / python.sh (Kit runtime).\n"
                f"Original error: {exc}"
            ) from exc

        # Resolve experience name -> absolute path when needed.
        exp_path = Path(launch.experience)
        if not exp_path.exists() and ("/" not in launch.experience) and ("\\" not in launch.experience):
            isaac_root = Path(os.environ.get("ISAACSIM_ROOT", r"C:\isaacsim"))
            candidate = isaac_root / "apps" / launch.experience
            if candidate.exists():
                exp_path = candidate
        experience_to_launch = str(exp_path) if exp_path.exists() else launch.experience

        sim_config: dict[str, Any] = {
            "headless": launch.headless,
            "width": launch.width,
            "height": launch.height,
        }
        if launch.renderer is not None:
            sim_config["renderer"] = launch.renderer

        # Use a lighter experience by default to reduce extension load surface.
        # Users can override via config["experience"].
        simulation_app = SimulationApp(sim_config, experience=experience_to_launch)

        # Now it is safe to import omni/isaac modules.
        import omni.kit.commands  # type: ignore[import-not-found]

        from isaacsim.core.api.world import World  # type: ignore[import-not-found]
        from isaacsim.core.prims import SingleArticulation  # type: ignore[import-not-found]
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore[import-not-found]

        self._enable_extension_best_effort("isaacsim.asset.importer.urdf")

        isaac = _IsaacHandles(
            SimulationApp=SimulationApp,
            World=World,
            SingleArticulation=SingleArticulation,
            ArticulationAction=ArticulationAction,
            omni_kit_commands=omni.kit.commands,
        )
        return simulation_app, isaac

    def _enable_extension_best_effort(self, ext_name: str) -> None:
        try:
            import omni.kit.app  # type: ignore[import-not-found]

            ext_mgr = omni.kit.app.get_app().get_extension_manager()
            ext_mgr.set_extension_enabled_immediate(ext_name, True)
        except Exception:
            return

    def _create_world(self, isaac: _IsaacHandles):
        World = isaac.World
        if World.instance():
            World.instance().clear_instance()
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()

        # Recommended Isaac Sim standalone init path: ensure sim context exists.
        try:
            if hasattr(world, "initialize_simulation_context_async"):
                self._simulation_app.run_coroutine(world.initialize_simulation_context_async())
        except Exception:
            pass
        return world

    def _apply_joint_damping(self, damping: float) -> None:
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import UsdPhysics  # type: ignore[import-not-found]

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return
            prim_paths: list[str] = []
            runtimes = getattr(self, "_robot_runtimes", None)
            if isinstance(runtimes, dict) and runtimes:
                prim_paths = [runtime.prim_path for runtime in runtimes.values()]
            elif getattr(self, "_robot_prim_path", None):
                prim_paths = [str(self._robot_prim_path)]
            for root_path in prim_paths:
                root = stage.GetPrimAtPath(root_path)
                if root is None or not root.IsValid():
                    continue
                for prim in stage.Traverse():
                    if not str(prim.GetPath()).startswith(root_path):
                        continue
                    if not prim.IsA(UsdPhysics.RevoluteJoint) and not prim.IsA(UsdPhysics.PrismaticJoint):
                        continue
                    for drive_name in ("angular", "linear"):
                        drive = UsdPhysics.DriveAPI.Get(prim, drive_name)
                        if not drive:
                            drive = UsdPhysics.DriveAPI.Apply(prim, drive_name)
                        drive.GetDampingAttr().Set(float(damping))
        except Exception as exc:
            self._warnings.append(f"Isaac joint damping tuning unavailable: {exc}")

    def _import_usd_robot(self, usd_path: Path, *, prim_path: str | None = None) -> str:
        prim_path = prim_path or f"/World/{self._robot_id}"
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore[import-not-found]

            add_reference_to_stage(usd_path=str(usd_path), prim_path=prim_path)
            return prim_path
        except Exception:
            pass
        try:
            import omni.usd  # type: ignore[import-not-found]

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("No active USD stage for robot import.")
            prim = stage.DefinePrim(prim_path, "Xform")
            prim.GetReferences().AddReference(str(usd_path).replace("\\", "/"))
            return prim_path
        except Exception as exc:
            raise RuntimeError(f"Failed to import USD robot from {usd_path}") from exc

    def _read_usd_prop_pose(self, name: str):
        path = getattr(self, "_scene_prop_paths", {}).get(name)
        if not path:
            return None
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import UsdGeom  # type: ignore[import-not-found]

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(path) if stage is not None else None
            if prim is None or not prim.IsValid():
                return None
            xform = UsdGeom.Xformable(prim)
            matrix = xform.ComputeLocalToWorldTransform(0)
            tr = matrix.ExtractTranslation()
            _, quat = self._scene_prop_poses[name]
            return (tuple(float(v) for v in tr), quat)
        except Exception:
            return None

    def _write_usd_prop_pose(self, name: str, position, orientation=None) -> bool:  # noqa: ANN001
        path = getattr(self, "_scene_prop_paths", {}).get(name)
        if not path:
            return False
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(path) if stage is not None else None
            if prim is None or not prim.IsValid():
                return False
            self._set_usd_pose(
                UsdGeom.XformCommonAPI(prim),
                Gf,
                position,
                orientation,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _set_usd_pose(xform_api, Gf, position, orientation=None) -> None:  # noqa: ANN001,N803
        xform_api.SetTranslate(Gf.Vec3d(*[float(v) for v in position]))
        if orientation is not None and hasattr(xform_api, "SetRotate"):
            xform_api.SetRotate(_quat_to_xyz_degrees(tuple(float(v) for v in orientation)))

    def _import_urdf_robot(self, isaac: _IsaacHandles, urdf_path: Path, *, prim_path: str | None = None) -> str:
        # URDF import config + import.
        try:
            from isaacsim.asset.importer.urdf import _urdf  # type: ignore[import-not-found]

            import_config = _urdf.ImportConfig()
            import_config.fix_base = bool(self.config.get("fix_base", True))
            import_config.make_default_prim = True
            import_config.self_collision = bool(self.config.get("self_collision", False))
            import_config.convex_decomp = bool(self.config.get("convex_decomp", False))
            import_config.import_inertia_tensor = bool(self.config.get("import_inertia_tensor", True))
            import_config.distance_scale = float(self.config.get("distance_scale", 1.0))
            import_config.density = float(self.config.get("density", 0.0))
        except Exception as exc:
            raise ImportError(
                "IsaacSimBackend could not import the URDF importer extension.\n"
                "Make sure the `isaacsim.asset.importer.urdf` extension is available/enabled.\n"
                f"Original error: {exc}"
            ) from exc

        ok, robot_model = isaac.omni_kit_commands.execute(
            "URDFParseFile",
            urdf_path=str(urdf_path),
            import_config=import_config,
        )
        if not ok:
            raise RuntimeError(f"URDFParseFile failed for {urdf_path}")

        drive_strength = float(self.config.get("drive_strength", 1047.19751))
        drive_damping = float(self.config.get("drive_damping", 52.35988))
        try:
            for joint_name in robot_model.joints:
                robot_model.joints[joint_name].drive.strength = drive_strength
                robot_model.joints[joint_name].drive.damping = drive_damping
        except Exception:
            pass

        import_kwargs: dict[str, Any] = {
            "urdf_robot": robot_model,
            "import_config": import_config,
        }
        if prim_path is not None:
            import_kwargs["dest_path"] = prim_path
        ok, imported_path = isaac.omni_kit_commands.execute("URDFImportRobot", **import_kwargs)
        if not ok:
            raise RuntimeError(f"URDFImportRobot failed for {urdf_path}")
        return str(imported_path or prim_path or f"/World/{self._robot_id}")

    def _ensure_physics_ready(self, world, simulation_app: _SimulationAppLike) -> None:
        # Ensure physics is ready before initializing articulations.
        try:
            if hasattr(world, "reset_async"):
                simulation_app.run_coroutine(world.reset_async())
            else:
                world.reset()
        except Exception:
            world.reset()

        # Some Isaac setups need an explicit physics init after importing prims.
        try:
            if hasattr(world, "initialize_physics"):
                world.initialize_physics()
        except Exception:
            pass

        # Start physics so articulation views can be created.
        try:
            if hasattr(world, "play_async"):
                simulation_app.run_coroutine(world.play_async())
            elif hasattr(world, "play"):
                world.play()
        except Exception:
            pass

        for _ in range(int(self.config.get("startup_frames", 60))):
            simulation_app.update()

    def _initialize_articulation(self, robot) -> None:
        try:
            robot.initialize()
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Isaac articulation. This usually means physics "
                "did not finish initializing (simulation view is None) or the imported "
                "URDF is missing articulation properties.\n"
                "\n"
                "On Windows, a very common root cause is a missing Microsoft Visual C++ "
                "runtime dependency, which causes Isaac extensions like "
                "`omni.physx.tensors` to fail to load (you'll see a log like "
                "\"dependent library failed to load\" for `omni.physx.tensors.plugin.dll`).\n"
                "Fix: install the latest \"Microsoft Visual C++ Redistributable 2015-2022 (x64)\" "
                "and re-run from `C:\\isaacsim\\python.bat`.\n"
                f"prim_path={getattr(self, '_robot_prim_path', '<unknown>')}\n"
                f"Original error: {exc}"
            ) from exc

    def _init_timing(self) -> None:
        self._sim_time = 0.0
        self._physics_dt = (
            float(self._world.get_physics_dt()) if hasattr(self._world, "get_physics_dt") else 1.0 / 60.0
        )
        self._steps_per_control = max(1, int(round((1.0 / float(self.control_hz)) / self._physics_dt)))

    def _seed_home_pose(self, description: "RobotDescription", *, headless: bool) -> None:
        home = getattr(description, "home_qpos", None)
        if home is None:
            return
        import numpy as np

        q = np.asarray(home, dtype=np.float32)
        self._robot.apply_action(self._isaac.ArticulationAction(joint_positions=q))
        for _ in range(5 * self._steps_per_control):
            self._world.step(render=not headless)
            self._simulation_app.update()

    def _reset_impl(self) -> Observation:
        headless = bool(self.config.get("headless", True))
        self._ensure_physics_ready(self._world, self._simulation_app)
        try:
            self._robot.initialize()
        except Exception as exc:
            raise RuntimeError("Failed to reinitialize Isaac articulation during reset.") from exc
        self._sim_time = 0.0

        home = getattr(self._description, "home_qpos", None)
        if home is not None:
            import numpy as np

            q = np.asarray(home, dtype=np.float32)
            self._robot.apply_action(self._isaac.ArticulationAction(joint_positions=q))
            for _ in range(5 * self._steps_per_control):
                self._world.step(render=not headless)
                self._simulation_app.update()

        obs = self._merge_sensor_data(self._build_obs(), self._sensors)
        if getattr(self, "_rviz", None) is not None:
            self._rviz.reset()
            self._rviz.publish_robot_state(self._robot_id, obs)
            if self._latest_viz_payload is not None:
                self._rviz.publish_task_viz(self._latest_viz_payload)
        return obs

    def _step_impl(self, action: Action) -> Observation:
        headless = bool(self.config.get("headless", True))

        if action.joint_positions is not None:
            import numpy as np

            q = np.asarray(action.joint_positions, dtype=np.float32)
            self._robot.apply_action(self._isaac.ArticulationAction(joint_positions=q))

        for _ in range(self._steps_per_control):
            self._world.step(render=not headless)
            self._simulation_app.update()
            self._sim_time += self._physics_dt

        obs = self._merge_sensor_data(self._build_obs(), self._sensors)
        if getattr(self, "_rviz", None) is not None:
            self._rviz.publish_robot_state(self._robot_id, obs)
            if self._latest_viz_payload is not None:
                self._rviz.publish_task_viz(self._latest_viz_payload)
        return obs

    def _get_obs_impl(self) -> Observation:
        return self._merge_sensor_data(self._build_obs(), self._sensors)

    def _close_impl(self) -> None:
        if getattr(self, "_rviz", None) is not None:
            try:
                self._rviz.close()
            except Exception:
                pass
        try:
            if hasattr(self, "_world") and self._world is not None:
                try:
                    self._world.pause()
                except Exception:
                    pass
        finally:
            try:
                if hasattr(self, "_simulation_app") and self._simulation_app is not None:
                    self._simulation_app.close()
            except Exception:
                pass

    # Optional hook for RoboEnv to provide task-goal visualization payload.
    def set_viz_payload(self, payload: Optional[dict]) -> None:
        self._latest_viz_payload = payload

    def _build_obs(self, *, robot=None, description=None) -> Observation:
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        import numpy as np

        articulation = robot or self._robot
        qpos = np.asarray(articulation.get_joint_positions(), dtype=np.float32)
        qvel = np.asarray(articulation.get_joint_velocities(), dtype=np.float32)
        dof = int(qpos.shape[0]) if qpos.ndim == 1 else int(len(qpos))

        qfrc = self._read_joint_efforts(dof, robot=articulation)
        ee_pos, ee_quat, ee_vel, ee_avel = self._read_ee_state(robot=articulation, description=description)

        return Observation(
            joint_positions=jnp.asarray(qpos, dtype=jnp.float32),
            joint_velocities=jnp.asarray(qvel, dtype=jnp.float32),
            joint_torques=jnp.asarray(qfrc, dtype=jnp.float32),
            ee_position=jnp.asarray(ee_pos, dtype=jnp.float32),
            ee_orientation=jnp.asarray(ee_quat, dtype=jnp.float32),
            ee_velocity=jnp.asarray(ee_vel, dtype=jnp.float32),
            ee_angular_velocity=jnp.asarray(ee_avel, dtype=jnp.float32),
            timestamp=float(self._sim_time),
            timestamp_hw=float(self._sim_time),
            timestamp_recv=float(self._sim_time),
        )

    def _read_joint_efforts(self, dof: int, *, robot=None):
        import numpy as np

        robot = robot or self._robot
        for target in (robot, getattr(robot, "_articulation_view", None), getattr(self, "_art_view", None)):
            if target is None:
                continue
            fn = getattr(target, "get_measured_joint_efforts", None)
            if callable(fn):
                try:
                    raw = fn()
                    if hasattr(raw, "cpu"):
                        raw = raw.cpu().numpy()
                    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
                    if arr.shape[0] >= dof:
                        return arr[:dof]
                except Exception:
                    continue
        if not any("joint efforts unavailable" in w for w in self._warnings):
            self._warnings.append("Isaac joint efforts unavailable; returning zeros.")
        return np.zeros((dof,), dtype=np.float32)

    @staticmethod
    def _isaac_articulation_targets(robot) -> list:
        targets: list = []
        seen: set[int] = set()
        for candidate in (robot, getattr(robot, "_articulation_view", None)):
            if candidate is None or id(candidate) in seen:
                continue
            targets.append(candidate)
            seen.add(id(candidate))
        return targets

    @staticmethod
    def _isaac_tensor_to_numpy(raw):
        import numpy as np

        if hasattr(raw, "cpu"):
            raw = raw.cpu().numpy()
        return np.asarray(raw, dtype=np.float32)

    @classmethod
    def _isaac_vec3(cls, raw):
        import numpy as np

        arr = cls._isaac_tensor_to_numpy(raw)
        if arr.ndim >= 2:
            arr = arr[0]
        return arr.reshape(-1)[:3].astype(np.float32, copy=False)

    @classmethod
    def _isaac_vec4(cls, raw):
        import numpy as np

        arr = cls._isaac_tensor_to_numpy(raw)
        if arr.ndim >= 2:
            arr = arr[0]
        return arr.reshape(-1)[:4].astype(np.float32, copy=False)

    def _ee_link_name_candidates(self, ee_link: str, description=None) -> list[str]:
        candidates: list[str] = []
        for name in (ee_link, str(self.config.get("ee_link", ""))):
            if name and name not in candidates:
                candidates.append(name)
        if description is not None and hasattr(description, "ros_ee_frame_id"):
            ros_name = description.ros_ee_frame_id()
            if ros_name and ros_name not in candidates:
                candidates.append(ros_name)
        for name in list(candidates):
            if "/" in name:
                short = name.split("/", 1)[1]
                if short not in candidates:
                    candidates.append(short)
        return candidates

    def _resolve_ee_link_index(self, targets: list, candidates: list[str]):
        for target in targets:
            for name in candidates:
                for attr in ("get_link_index", "get_body_index"):
                    fn = getattr(target, attr, None)
                    if not callable(fn):
                        continue
                    try:
                        return int(fn(name))
                    except Exception:
                        continue
        for target in targets:
            for attr in ("num_links", "num_bodies"):
                count = getattr(target, attr, None)
                if isinstance(count, int) and count > 0:
                    return count - 1
            names = getattr(target, "link_names", None) or getattr(target, "body_names", None)
            if names:
                try:
                    n = len(list(names))
                except Exception:
                    n = 0
                if n > 0:
                    return n - 1
        return None

    def _read_ee_state(self, *, robot=None, description=None):
        import numpy as np

        ee_pos = np.zeros((3,), dtype=np.float32)
        ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ee_vel = np.zeros((3,), dtype=np.float32)
        ee_avel = np.zeros((3,), dtype=np.float32)
        warnings = getattr(self, "_warnings", None)
        if warnings is None:
            warnings = []
            self._warnings = warnings

        description = description if description is not None else getattr(self, "_description", None)
        ee_link = None
        if description is not None:
            ee_link = getattr(description, "ee_link_name", None)
        if not ee_link:
            ee_link = str(self.config.get("ee_link", "ee_link"))

        articulation = robot or self._robot
        targets = self._isaac_articulation_targets(articulation)
        if getattr(self, "_art_view", None) is not None and self._art_view not in targets:
            targets.append(self._art_view)

        link_idx = self._resolve_ee_link_index(targets, self._ee_link_name_candidates(ee_link, description))
        if link_idx is None:
            if not any("ee state unavailable" in w for w in warnings):
                warnings.append("Isaac EE state unavailable; returning zeros.")
            return ee_pos, ee_quat, ee_vel, ee_avel

        pose_read = False
        for target in targets:
            for pose_attr in ("get_link_world_poses", "get_world_poses"):
                pose_fn = getattr(target, pose_attr, None)
                if not callable(pose_fn):
                    continue
                try:
                    poses = pose_fn(indices=[link_idx])
                    if isinstance(poses, tuple) and len(poses) >= 2:
                        positions, orientations = poses[0], poses[1]
                    else:
                        positions = getattr(poses, "positions", poses)
                        orientations = getattr(poses, "orientations", None)
                    ee_pos = self._isaac_vec3(positions)
                    if orientations is not None:
                        ee_quat = self._isaac_vec4(orientations)
                    pose_read = True
                    break
                except Exception:
                    continue
            if pose_read:
                break

        for target in targets:
            vel_fn = getattr(target, "get_link_velocities", None)
            if not callable(vel_fn):
                vel_fn = getattr(target, "get_velocities", None)
            if not callable(vel_fn):
                continue
            try:
                lin_ang = vel_fn(indices=[link_idx])
                if isinstance(lin_ang, tuple) and len(lin_ang) >= 2:
                    ee_vel = self._isaac_vec3(lin_ang[0])
                    ee_avel = self._isaac_vec3(lin_ang[1])
                else:
                    combined = self._isaac_tensor_to_numpy(lin_ang).reshape(-1)
                    if combined.size < 6:
                        continue
                    ee_vel = combined[:3].astype(np.float32, copy=False)
                    ee_avel = combined[3:6].astype(np.float32, copy=False)
                break
            except Exception:
                continue

        if not pose_read and float(np.linalg.norm(ee_pos)) == 0.0:
            if not any("ee state unavailable" in w for w in warnings):
                warnings.append("Isaac EE pose unavailable; returning zeros.")
        return ee_pos, ee_quat, ee_vel, ee_avel

