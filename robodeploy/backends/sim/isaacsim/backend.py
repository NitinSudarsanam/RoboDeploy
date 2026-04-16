"""IsaacSimBackend — optional Isaac Sim simulation backend.

This module isolates Isaac imports so the library remains importable on
machines that do not have Isaac Sim installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol

from robodeploy.backends.base import BackendBase
from robodeploy.core.registry import register_backend
from robodeploy.core.spaces import ActionSpace, AssetFormat
from robodeploy.core.types import Action, Observation

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.core.interfaces.task import ITask
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


@register_backend("isaacsim")
class IsaacSimBackend(BackendBase):
    """Isaac Sim backend (optional; runtime-guarded)."""

    is_real = False
    control_hz = 100.0
    supported_action_spaces = [ActionSpace.JOINT_POS]

    def _load(self, description: RobotDescription, task: ITask, sensors: list[ISensor]) -> None:
        del task, sensors
        launch = self._parse_launch_config()
        self._simulation_app, self._isaac = self._launch_kit(launch)
        self._world = self._create_world(self._isaac)

        urdf_path = self._resolve_asset_path("robot0", description, AssetFormat.URDF, variant="sim")
        self._robot_prim_path = self._import_urdf_robot(self._isaac, urdf_path)
        self._robot = self._isaac.SingleArticulation(prim_path=self._robot_prim_path, name="robot0")

        self._ensure_physics_ready(self._world, self._simulation_app)
        self._initialize_articulation(self._robot)

        self._init_timing()
        self._seed_home_pose(description, headless=launch.headless)

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

    def _import_urdf_robot(self, isaac: _IsaacHandles, urdf_path: Path) -> str:
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

        ok, prim_path = isaac.omni_kit_commands.execute(
            "URDFImportRobot",
            urdf_robot=robot_model,
            import_config=import_config,
        )
        if not ok:
            raise RuntimeError(f"URDFImportRobot failed for {urdf_path}")
        return str(prim_path)

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
            try:
                simulation_app.update()
            except Exception:
                break

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
        except Exception:
            return self._build_obs()
        self._sim_time = 0.0

        home = getattr(self._description, "home_qpos", None)
        if home is not None:
            import numpy as np

            q = np.asarray(home, dtype=np.float32)
            self._robot.apply_action(self._isaac.ArticulationAction(joint_positions=q))
            for _ in range(5 * self._steps_per_control):
                self._world.step(render=not headless)
                self._simulation_app.update()

        return self._build_obs()

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

        return self._build_obs()

    def _get_obs_impl(self) -> Observation:
        return self._build_obs()

    def _close_impl(self) -> None:
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

    def _build_obs(self) -> Observation:
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        import numpy as np

        qpos = np.asarray(self._robot.get_joint_positions(), dtype=np.float32)
        qvel = np.asarray(self._robot.get_joint_velocities(), dtype=np.float32)
        dof = int(qpos.shape[0]) if qpos.ndim == 1 else int(len(qpos))

        # Isaac doesn't always expose efforts consistently across assets; default to zeros.
        qfrc = np.zeros((dof,), dtype=np.float32)

        # End-effector: best-effort (keep shapes stable even if lookup fails).
        ee_pos = np.zeros((3,), dtype=np.float32)
        ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ee_vel = np.zeros((3,), dtype=np.float32)
        ee_avel = np.zeros((3,), dtype=np.float32)

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

