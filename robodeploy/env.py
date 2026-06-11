"""RoboEnv — robot-centric orchestrator.

Constructed with a backend and a list of `Robot` aggregates. Each `Robot`
encapsulates its own description, sensors, tasks, policies, and arbitration
choice. RoboEnv steps every robot, evaluates rewards/success per active task,
and merges the per-robot observations / states into a single EpisodeInfo.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from robodeploy.backends.capabilities import (
    SupportsContactQuery,
    SupportsDiagnostics,
    SupportsMultiRobot,
    SupportsVizSink,
)
from robodeploy.core.extra_schemas import (
    build_assets_extra,
    build_diagnostics_extra,
    build_multi_agent_extra,
    build_viz_extra,
)
from robodeploy.core.interfaces.backend import IBackend
from robodeploy.core.interfaces.policy import IPolicy
from robodeploy.core.interfaces.sensor import ISensor
from robodeploy.core.interfaces.task import ITask
from robodeploy.core.registry import (
    get_backend,
    get_policy,
    get_robot,
    get_task,
    normalize_sensor_backend_name,
    resolve_sensor_class,
)
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import (
    Action,
    EpisodeInfo,
    HumanInterventionRequired,
    MultiAgentInfo,
    Observation,
    Pose3D,
    PropConfig,
    RobotStepState,
    SceneSpec,
    TaskStepState,
)
from robodeploy.core.seeding import SeedSet, derive_seeds, seed_global_rngs, seedset_as_dict
from robodeploy.obs_pipeline import ObsPipeline
from robodeploy.observability.health import HealthMonitor, summarize_sensor_health
from robodeploy.safety import (
    CollisionGuard,
    ForceLimitGuard,
    SafetyError,
    SafetyFilterGuard,
    SafetyMonitor,
    VelocityGuard,
)
from robodeploy.safety.monitor import SafetyStatus


class RoboEnv:
    """Robot-centric env. One backend, many robots, one step path."""

    def __init__(
        self,
        backend: IBackend,
        robots: List[Robot],
        *,
        shared_sensors: Optional[List[ISensor]] = None,
        max_episode_steps: Optional[int] = None,
        obs_spec_policy: str = "warn",
        logger=None,
        health_monitor: Optional[HealthMonitor] = None,
        record_manifest: bool = False,
        run_name: Optional[str] = None,
        env_config: Optional[dict] = None,
        safety: SafetyMonitor | None = None,
        safety_enabled: bool = True,
        safety_config: Optional[dict] = None,
        emergency_action: Action | None = None,
    ) -> None:
        if backend is None:
            raise ValueError("RoboEnv requires a backend.")
        if not robots:
            raise ValueError("RoboEnv requires at least one Robot.")

        self._backend = backend
        self._robots: List[Robot] = list(robots)
        self._robot_by_id: Dict[str, Robot] = {r.robot_id: r for r in self._robots}
        if len(self._robot_by_id) != len(self._robots):
            raise ValueError("Robot IDs must be unique within a RoboEnv.")

        self._shared_sensors: List[ISensor] = shared_sensors or []
        self._obs_spec_policy = str(obs_spec_policy)
        self._episode_info = EpisodeInfo()
        self._initialized = False
        self._on_pause: Optional[Callable[[], None]] = None
        self._on_resume: Optional[Callable[[], None]] = None
        self._on_intervention: Optional[Callable[[HumanInterventionRequired], None]] = None
        self._logger = logger
        self._health_monitor = health_monitor or HealthMonitor()
        self._record_manifest = bool(record_manifest)
        self._run_name = run_name
        self._manifest_recorder = None
        self._master_seed: Optional[int] = None
        self._seeds: Optional[SeedSet] = None
        self.env_config_snapshot = dict(env_config or {})
        self.seed_snapshot: dict[str, int] = {}
        self._emergency_action = emergency_action
        self._last_obs_by_robot: Dict[str, Observation] = {}
        self._safety_enabled = bool(safety_enabled)
        self._safety_config = dict(safety_config or {})
        self._policy_diagnostics: dict[str, Any] = {}
        self._negotiate_action_spaces()
        if safety is not None:
            self._safety = safety
        elif self._safety_enabled:
            self._safety = self._build_default_safety()
        else:
            self._safety = None
        if self._safety is not None:
            from robodeploy.safety.registry import register_safety_monitor

            label = ",".join(robot.robot_id for robot in self._robots)
            register_safety_monitor(self._safety, label=label)

        primary_robot = self._robots[0]
        primary_task_id = primary_robot.active_task_id
        primary_task = primary_robot.tasks[primary_task_id] if primary_task_id else None
        self._max_steps = max_episode_steps or (primary_task.task.max_steps() if primary_task else 1000)

    # ------------------------------------------------------------------
    # Construction sugar
    # ------------------------------------------------------------------

    @classmethod
    def make(
        cls,
        robot: str,
        backend: str,
        task: str,
        policy: Optional[str] = None,
        sensors: Optional[List[str]] = None,
        sensor_kwargs: Optional[dict] = None,
        sensor_rigs: Optional[List[Any]] = None,
        backend_kwargs: Optional[dict] = None,
        task_kwargs: Optional[dict] = None,
        policy_kwargs: Optional[dict] = None,
        obs_pipeline: Optional[ObsPipeline] = None,
        obs_pipeline_spec: Optional[dict] = None,
        custom_modules: Optional[List[str]] = None,
        obs_spec_policy: str = "warn",
        max_episode_steps: Optional[int] = None,
        robot_id: str = "robot0",
        task_id: str = "task0",
        policy_id: str = "policy0",
    ) -> "RoboEnv":
        if policy is None:
            raise ValueError(
                "RoboEnv.make() requires a policy name. "
                "Construct RoboEnv(robots=[...]) directly to use external action injection."
            )
        cfg: dict[str, Any] = {
            "robot": robot,
            "backend": backend,
            "task": task,
            "policy": policy,
            "robot_id": robot_id,
            "task_id": task_id,
            "policy_id": policy_id,
            "obs_spec_policy": obs_spec_policy,
        }
        if sensors:
            cfg["sensors"] = list(sensors)
        if sensor_kwargs:
            cfg["sensor_kwargs"] = sensor_kwargs
        if sensor_rigs:
            cfg["sensor_rigs"] = sensor_rigs
        if backend_kwargs:
            cfg["backend_kwargs"] = backend_kwargs
        if task_kwargs:
            cfg["task_kwargs"] = task_kwargs
        if policy_kwargs:
            cfg["policy_kwargs"] = policy_kwargs
        if custom_modules:
            cfg["custom_modules"] = list(custom_modules)
        if obs_pipeline_spec:
            cfg["obs_pipeline"] = obs_pipeline_spec
        if max_episode_steps is not None:
            cfg["max_episode_steps"] = max_episode_steps
        return cls.from_config(cfg, obs_pipeline=obs_pipeline)

    @classmethod
    def from_config(
        cls,
        cfg: dict | Any,
        obs_pipeline: Optional[ObsPipeline] = None,
    ) -> "RoboEnv":
        from robodeploy.builtins import import_builtins
        from robodeploy.core.env_config import EnvConfig
        from robodeploy.core.registry import use

        import_builtins()
        if isinstance(cfg, EnvConfig):
            cfg = cfg.to_dict()
        cfg = dict(cfg)
        for module_path in cfg.pop("custom_modules", []):
            use(module_path)

        backend_value = cfg["backend"]
        backend_kwargs = cfg.get("backend_kwargs")
        sensor_ctx = (
            cls._backend_sensor_context(backend_value)
            if isinstance(backend_value, str)
            else cls._coerce_backend(backend_value, backend_kwargs)
        )
        sensor_backend_name = cls._sensor_backend_name_for(
            sensor_ctx,
            default_name=backend_value if isinstance(backend_value, str) else None,
        )
        if "robots" in cfg:
            robots = cls._coerce_robots_list(
                cfg["robots"],
                backend_obj=sensor_ctx,
                sensor_backend_name=sensor_backend_name,
                default_obs_pipeline=obs_pipeline,
                cfg=cfg,
            )
            shared_raw = cfg.get("shared_sensors")
            shared_sensors = (
                cls._coerce_sensors(
                    shared_raw,
                    cfg.get("sensor_kwargs"),
                    is_real=sensor_ctx.is_real,
                    backend_name=sensor_backend_name,
                )
                if shared_raw is not None
                else None
            )
            backend_obj = cls._build_backend(backend_value, backend_kwargs, robots, cfg)
            return cls(
                backend=backend_obj,
                robots=robots,
                shared_sensors=shared_sensors,
                max_episode_steps=cfg.get("max_episode_steps"),
                obs_spec_policy=str(cfg.get("obs_spec_policy", "warn")),
                safety_config=cfg.get("safety"),
            )

        robot_value = cfg["robot"]
        if isinstance(robot_value, Robot):
            backend_obj = cls._build_backend(backend_value, backend_kwargs, [robot_value], cfg)
            return cls(
                backend=backend_obj,
                robots=[robot_value],
                shared_sensors=cfg.get("shared_sensors"),
                max_episode_steps=cfg.get("max_episode_steps"),
                safety_config=cfg.get("safety"),
            )

        description_obj = cls._coerce_description(robot_value, cfg.get("robot_kwargs"))
        task_obj = cls._coerce_task(cfg["task"], cfg.get("task_kwargs"))
        policy_value = cfg.get("policy")
        if policy_value is None:
            raise ValueError("RoboEnv.from_config() requires a policy unless cfg['robot'] is already a Robot.")
        policy_obj = cls._coerce_policy(policy_value, cfg.get("policy_kwargs"))
        sensor_objs = cls._coerce_sensors(
            cfg.get("sensors"),
            cfg.get("sensor_kwargs"),
            is_real=sensor_ctx.is_real,
            backend_name=sensor_backend_name,
        )
        rig_specs = cfg.get("sensor_rigs")
        if rig_specs:
            from robodeploy.core.sensor_rig import SensorRig, materialize_sensor_rigs

            rigs: list[SensorRig] = []
            for entry in rig_specs:
                if isinstance(entry, SensorRig):
                    rigs.append(entry)
                elif isinstance(entry, dict):
                    rigs.append(
                        SensorRig.robot_mounted(
                            str(entry.get("rig_id", "arm_sensors")),
                            ee_link=str(entry.get("ee_link", "robot0/ee_link")),
                            wrist_rgbd=entry.get("wrist_rgbd"),
                            overhead_rgbd=entry.get("overhead_rgbd"),
                            wrist_ft=entry.get("wrist_ft"),
                            wrist_imu=entry.get("wrist_imu"),
                            base_imu=entry.get("base_imu"),
                            wrist_contact=entry.get("wrist_contact"),
                            prop_pose=entry.get("prop_pose"),
                        )
                    )
                else:
                    raise TypeError("sensor_rigs entries must be SensorRig instances or dicts.")
            sensor_objs = list(sensor_objs) + materialize_sensor_rigs(
                rigs,
                is_real=sensor_ctx.is_real,
                backend_name=sensor_backend_name,
            )

        pipeline = obs_pipeline or cls._coerce_obs_pipeline(cfg.get("obs_pipeline"))
        robot_obj = Robot(
            robot_id=str(cfg.get("robot_id", "robot0")),
            description=description_obj,
            tasks={
                str(cfg.get("task_id", "task0")): RobotTask(
                    task=task_obj,
                    policies={str(cfg.get("policy_id", "policy0")): policy_obj},
                    task_id=str(cfg.get("task_id", "task0")),
                ),
            },
            sensors=sensor_objs,
            obs_pipeline=pipeline,
        )
        backend_obj = cls._build_backend(backend_value, backend_kwargs, [robot_obj], cfg)
        return cls(
            backend=backend_obj,
            robots=[robot_obj],
            shared_sensors=cfg.get("shared_sensors"),
            max_episode_steps=cfg.get("max_episode_steps"),
            obs_spec_policy=str(cfg.get("obs_spec_policy", "warn")),
            safety_config=cfg.get("safety"),
        )

    @staticmethod
    def _instantiate_component(value: Any, kwargs: Optional[dict]) -> Any:
        if isinstance(value, type):
            return value(**(kwargs or {}))
        if callable(value) and not isinstance(value, (str, Robot)):
            return value(**(kwargs or {}))
        return value

    @classmethod
    def _backend_sensor_context(cls, backend_name: str) -> Any:
        BackendClass = get_backend(backend_name)

        class _Ctx:
            is_real = bool(getattr(BackendClass, "is_real", False))
            sensor_backend_name = getattr(BackendClass, "sensor_backend_name", None)

        return _Ctx()

    @classmethod
    def _build_backend(
        cls,
        backend_value: Any,
        backend_kwargs: Optional[dict],
        robots: list[Robot],
        cfg: dict,
    ) -> IBackend:
        if not isinstance(backend_value, str):
            return cls._coerce_backend(backend_value, backend_kwargs)

        from robodeploy.backends.simulator import (
            backend_for_simulator,
            behavior_profile_from_config,
            normalize_backend_config_overrides,
            simulator_name_for_backend,
        )

        simulator = simulator_name_for_backend(backend_value)
        if simulator is None or not robots:
            return cls._coerce_backend(backend_value, backend_kwargs)

        raw_kwargs = dict(backend_kwargs or {})
        behavior = behavior_profile_from_config(cfg, raw_kwargs)
        local_ros_graph = raw_kwargs.pop("local_ros_graph", None)
        if local_ros_graph is None and simulator == "ros2_rviz":
            local_ros_graph = True

        config_overrides = normalize_backend_config_overrides(raw_kwargs)
        try:
            return backend_for_simulator(
                simulator,
                robots=robots,
                local_ros_graph=bool(local_ros_graph),
                config_overrides=config_overrides,
                behavior=behavior,
            )
        except ValueError as exc:
            if "Gazebo requires a sim world" in str(exc):
                return cls._coerce_backend(backend_value, backend_kwargs)
            raise

    @classmethod
    def _coerce_backend(cls, value: Any, kwargs: Optional[dict]) -> IBackend:
        if isinstance(value, str):
            BackendClass = get_backend(value)
            return BackendClass(**(kwargs or {}))
        obj = cls._instantiate_component(value, kwargs)
        if not isinstance(obj, IBackend):
            raise TypeError("backend must be a registry name, backend class, or IBackend instance.")
        return obj

    @classmethod
    def _coerce_description(cls, value: Any, kwargs: Optional[dict]):
        from robodeploy.description.base import RobotDescription

        if isinstance(value, str):
            DescriptionClass = get_robot(value)
            return DescriptionClass(**(kwargs or {}))
        obj = cls._instantiate_component(value, kwargs)
        if not isinstance(obj, RobotDescription):
            raise TypeError("robot must be a registry name, RobotDescription class, RobotDescription instance, or Robot.")
        return obj

    @classmethod
    def _coerce_task(cls, value: Any, kwargs: Optional[dict]) -> ITask:
        if isinstance(value, str):
            TaskClass = get_task(value)
            kw = dict(kwargs or {})
            try:
                return TaskClass(**kw)
            except TypeError:
                return TaskClass(config=kw)
        obj = cls._instantiate_component(value, kwargs)
        if not isinstance(obj, ITask):
            raise TypeError("task must be a registry name, task class, or ITask instance.")
        return obj

    @classmethod
    def _coerce_obs_pipeline(cls, spec: Any) -> ObsPipeline:
        if spec is None:
            return ObsPipeline()
        if isinstance(spec, ObsPipeline):
            return spec
        if not isinstance(spec, dict):
            raise TypeError("obs_pipeline must be an ObsPipeline instance or dict spec.")
        import importlib

        transforms = []
        for entry in spec.get("transforms", []):
            if not isinstance(entry, dict):
                raise TypeError("obs_pipeline.transforms entries must be dicts.")
            module = importlib.import_module(str(entry["module"]))
            transform_cls = getattr(module, str(entry["class"]))
            transforms.append(transform_cls(**dict(entry.get("kwargs") or {})))
        return ObsPipeline(transforms)

    @classmethod
    def _coerce_policy(cls, value: Any, kwargs: Optional[dict]) -> IPolicy:
        if isinstance(value, str):
            ref = str(value)
            if ref.startswith("hf:") or (":" in ref and not ref.startswith("http")) or ref.endswith((".pt", ".pth", ".ckpt")):
                from robodeploy.evaluation.policy_loader import coerce_eval_policy

                return coerce_eval_policy(ref, kwargs or {})
            PolicyClass = get_policy(ref)
            return PolicyClass(config=kwargs or {})
        obj = cls._instantiate_component(value, kwargs)
        if not isinstance(obj, IPolicy):
            raise TypeError("policy must be a registry name, policy class, or IPolicy instance.")
        return obj

    @staticmethod
    def _coerce_robot_object(value: Any) -> Robot:
        if not isinstance(value, Robot):
            raise TypeError("cfg['robots'] entries must be Robot instances.")
        return value

    @classmethod
    def _coerce_pose3d(cls, value: Any) -> Pose3D | None:
        if value is None:
            return None
        if isinstance(value, Pose3D):
            return value
        if not isinstance(value, dict):
            raise TypeError("base_pose must be a Pose3D or dict with position/orientation.")
        pos = tuple(float(v) for v in value.get("position", (0.0, 0.0, 0.0)))
        quat = tuple(float(v) for v in value.get("orientation", (1.0, 0.0, 0.0, 0.0)))
        return Pose3D(position=pos, orientation=quat)

    @classmethod
    def _coerce_robots_list(
        cls,
        entries: list[Any],
        *,
        backend_obj: IBackend,
        sensor_backend_name: str | None,
        default_obs_pipeline: ObsPipeline | None,
        cfg: dict,
    ) -> list[Robot]:
        robots: list[Robot] = []
        for idx, item in enumerate(entries):
            if isinstance(item, Robot):
                robots.append(item)
                continue
            if not isinstance(item, dict):
                raise TypeError("cfg['robots'] entries must be Robot instances or dict specs.")
            robot_id = str(item.get("robot_id", f"robot{idx}"))
            description = cls._coerce_description(item.get("robot") or item.get("description"), item.get("robot_kwargs"))
            task_id = str(item.get("task_id", "task0"))
            policy_id = str(item.get("policy_id", "policy0"))
            task_obj = cls._coerce_task(item["task"], {**(cfg.get("task_kwargs") or {}), **(item.get("task_kwargs") or {})})
            policy_obj = cls._coerce_policy(
                item["policy"],
                {**(cfg.get("policy_kwargs") or {}), **(item.get("policy_kwargs") or {})},
            )
            sensor_objs = cls._coerce_sensors(
                item.get("sensors") or cfg.get("sensors"),
                item.get("sensor_kwargs") or cfg.get("sensor_kwargs"),
                is_real=backend_obj.is_real,
                backend_name=sensor_backend_name,
            )
            rig_specs = item.get("sensor_rigs") or cfg.get("sensor_rigs")
            if rig_specs:
                from robodeploy.core.sensor_rig import SensorRig, materialize_sensor_rigs

                rigs: list[SensorRig] = []
                for entry in rig_specs:
                    if isinstance(entry, SensorRig):
                        rigs.append(entry)
                    elif isinstance(entry, dict):
                        rigs.append(
                            SensorRig.robot_mounted(
                                str(entry.get("rig_id", "arm_sensors")),
                                ee_link=str(entry.get("ee_link", f"{robot_id}/ee_link")),
                                wrist_rgbd=entry.get("wrist_rgbd"),
                                overhead_rgbd=entry.get("overhead_rgbd"),
                                wrist_ft=entry.get("wrist_ft"),
                                wrist_imu=entry.get("wrist_imu"),
                                base_imu=entry.get("base_imu"),
                                wrist_contact=entry.get("wrist_contact"),
                                prop_pose=entry.get("prop_pose"),
                            )
                        )
                    else:
                        raise TypeError("sensor_rigs entries must be SensorRig instances or dicts.")
                sensor_objs = list(sensor_objs) + materialize_sensor_rigs(
                    rigs,
                    is_real=backend_obj.is_real,
                    backend_name=sensor_backend_name,
                )
            pipeline = cls._coerce_obs_pipeline(item.get("obs_pipeline") or cfg.get("obs_pipeline"))
            if pipeline is None and default_obs_pipeline is not None:
                pipeline = default_obs_pipeline
            resolver = item.get("task_action_resolver")
            robots.append(
                Robot(
                    robot_id=robot_id,
                    description=description,
                    tasks={
                        task_id: RobotTask(
                            task=task_obj,
                            policies={policy_id: policy_obj},
                            task_id=task_id,
                            mode=str(item.get("task_mode", "sequential")),
                        ),
                    },
                    sensors=sensor_objs,
                    obs_pipeline=pipeline or ObsPipeline(),
                    base_pose=cls._coerce_pose3d(item.get("base_pose")),
                    task_action_resolver=resolver,
                )
            )
        return robots

    @classmethod
    def _coerce_sensors(
        cls,
        sensors: Optional[List[Any]],
        sensor_kwargs: Optional[dict],
        *,
        is_real: bool,
        backend_name: str | None = None,
    ) -> list[ISensor]:
        out: list[ISensor] = []
        for item in sensors or []:
            if isinstance(item, str):
                SensorClass = resolve_sensor_class(
                    item,
                    is_real=is_real,
                    backend_name=backend_name,
                )
                cfg = dict((sensor_kwargs or {}).get(item, {}) or {})
                try:
                    out.append(SensorClass(config=cfg))
                except TypeError:
                    out.append(SensorClass())
            else:
                sensor_obj = cls._instantiate_component(item, None)
                if not isinstance(sensor_obj, ISensor):
                    raise TypeError("sensors entries must be registry names, sensor classes, or ISensor instances.")
                out.append(sensor_obj)
        return out

    @staticmethod
    def _sensor_backend_name_for(backend: IBackend, default_name: str | None = None) -> str | None:
        name = getattr(backend, "sensor_backend_name", None)
        if name:
            return normalize_sensor_backend_name(str(name))
        return normalize_sensor_backend_name(default_name)

    # ------------------------------------------------------------------
    # External hooks
    # ------------------------------------------------------------------

    def set_pause_hooks(self, on_pause: Callable[[], None], on_resume: Callable[[], None]) -> None:
        self._on_pause = on_pause
        self._on_resume = on_resume

    def set_intervention_handler(self, handler: Callable[[HumanInterventionRequired], None]) -> None:
        self._on_intervention = handler

    def switch_task(self, robot_id: str, to_task_id: str, reason: str = "") -> None:
        if robot_id not in self._robot_by_id:
            raise KeyError(f"Unknown robot_id '{robot_id}'.")
        self._robot_by_id[robot_id].switch_task(to_task_id, reason=reason)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_real(self) -> bool:
        return self._backend.is_real

    @property
    def backend(self) -> IBackend:
        return self._backend

    @property
    def robots(self) -> List[Robot]:
        return list(self._robots)

    @property
    def primary_robot(self) -> Robot:
        return self._robots[0]

    @property
    def max_episode_steps(self) -> int:
        return int(self._max_steps)

    @property
    def safety_monitor(self) -> SafetyMonitor | None:
        return self._safety

    @property
    def master_seed(self) -> Optional[int]:
        return self._master_seed

    @property
    def logger(self):
        return self._logger

    @property
    def health_monitor(self) -> HealthMonitor:
        return self._health_monitor

    def _seed_components(self) -> None:
        if self._seeds is None:
            return
        seed_global_rngs(self._seeds.env_seed)
        self.seed_snapshot = seedset_as_dict(self._seeds)
        backend_seed = getattr(self._backend, "seed", None)
        if callable(backend_seed):
            backend_seed(self._seeds.env_seed)
        for robot in self._robots:
            for robot_task in robot.tasks.values():
                randomizer_fn = getattr(robot_task.task, "_domain_randomizer", None)
                if callable(randomizer_fn):
                    dr = randomizer_fn()
                    if dr is not None:
                        dr.seed(self._seeds.randomizer_seed)
                for transform in robot.obs_pipeline.transforms:
                    reseed = getattr(transform, "seed", None)
                    if callable(reseed):
                        reseed(self._seeds.obs_pipeline_seed)
                    elif hasattr(transform, "_rng"):
                        try:
                            import numpy as np

                            transform._rng = np.random.default_rng(self._seeds.obs_pipeline_seed)
                        except Exception:
                            pass
        for sensor in self._all_sensors():
            sensor.seed(self._seeds.sensor_noise_seed)

    def _robot_action_space(self, robot: Robot) -> "ActionSpace":
        from robodeploy.core.spaces import ActionSpace

        if robot.effective_action_space is not None:
            return robot.effective_action_space
        task_id = robot.active_task_id or next(iter(robot.tasks))
        return robot.tasks[task_id].action_space()

    def _build_default_safety(self) -> SafetyMonitor:
        cfg = self._safety_config
        guards: list = []
        for robot in self._robots:
            desc = robot.description
            action_space = self._robot_action_space(robot)
            guards.extend(
                [
                    SafetyFilterGuard(
                        safety_filter=desc.get_safety_filter(),
                        action_space=action_space,
                        robot_id=robot.robot_id,
                    ),
                    ForceLimitGuard(
                        max_force_N=float(cfg.get("max_force_N", 50.0)),
                        over_limit_strikes=int(cfg.get("over_limit_strikes", 3)),
                        robot_id=robot.robot_id,
                    ),
                    VelocityGuard(
                        max_joint_velocity=desc.joint_velocity_limits,
                        robot_id=robot.robot_id,
                    ),
                ]
            )
        if isinstance(self._backend, SupportsContactQuery):
            guards.append(CollisionGuard(backend=self._backend))
        return SafetyMonitor(guards=guards, on_violation="clamp", on_critical="raise")

    def emergency_stop(self, reason: str = "external") -> None:
        for robot in self._robots:
            robot.description.get_safety_filter().trigger_estop()
        if self._safety is not None:
            self._safety.estop.trip(reason)
            from robodeploy.safety.violation import Hazard

            self._safety.halt(reason, hazard=Hazard.OPERATOR_ESTOP)
        self._enter_safe_state()

    def reset_safety(self) -> None:
        for robot in self._robots:
            robot.description.get_safety_filter().clear_estop()
        if self._safety is not None:
            self._safety.reset()

    def _hold_actions_for_step_multi(
        self,
        *,
        active: dict[str, Action] | None = None,
    ) -> list[Action]:
        hold_actions: list[Action] = []
        active = active or {}
        for robot in self._robots:
            if robot.robot_id in active:
                hold_actions.append(active[robot.robot_id])
                continue
            if self._emergency_action is not None:
                hold_actions.append(self._emergency_action)
                continue
            last = self._last_obs_by_robot.get(robot.robot_id)
            if last is not None:
                hold_actions.append(Action(joint_positions=last.joint_positions))
            else:
                hold_actions.append(Action(joint_positions=robot.description.home_qpos))
        return hold_actions

    def _warm_start_action_adapters(self, raw_obs_list: list[Observation]) -> None:
        import numpy as np

        for idx, robot in enumerate(self._robots[: len(raw_obs_list)]):
            q = raw_obs_list[idx].joint_positions
            if q is None:
                continue
            robot.action_adapter.warm_start(np.asarray(q, dtype=np.float64))

    def _enter_safe_state(self) -> None:
        if not self._initialized:
            return
        try:
            self._backend.step_multi(self._hold_actions_for_step_multi())
        except Exception:
            pass

    def _build_safety_info(self, err: SafetyError) -> EpisodeInfo:
        info = EpisodeInfo(
            episode_id=self._episode_info.episode_id,
            step=self._episode_info.step,
            reward=0.0,
            success=False,
            failure=True,
        )
        info.extra["safety"] = self._safety_payload(err)
        return info

    def _safety_payload(self, err: SafetyError | None = None) -> dict:
        status: SafetyStatus | None = self._safety.status() if self._safety is not None else None
        payload: dict = {
            "tripped": bool(status.tripped) if status else False,
            "history_count": int(status.history_count) if status else 0,
        }
        if status and status.last_violation is not None:
            v = status.last_violation
            payload["last_violation"] = {
                "hazard": v.hazard.name,
                "severity": v.severity.name,
                "message": v.message,
            }
        if err is not None:
            payload["error"] = str(err)
            payload["hazard"] = err.violation.hazard.name
        return payload

    def _control_dt(self) -> float:
        hz = float(getattr(self._backend, "control_hz", 0.0) or 0.0)
        return 1.0 / hz if hz > 0 else 0.05

    # ------------------------------------------------------------------
    # Initialization & reset
    # ------------------------------------------------------------------

    def _merged_scene(self) -> SceneSpec:
        merged = SceneSpec()
        seen_props: dict[str, PropConfig] = {}
        seen_cameras: set[str] = set()
        for robot in self._robots:
            for robot_task in robot.tasks.values():
                scene = robot_task.task.scene_spec()
                world = scene.to_world()
                for prop in world.props:
                    existing = seen_props.get(prop.name)
                    if existing is not None:
                        if existing != prop:
                            raise ValueError(f"Scene prop name collision for '{prop.name}'.")
                        continue
                    merged.world.props.append(prop)
                    seen_props[prop.name] = prop
                for light in world.lights:
                    merged.world.lights.append(light)
                for camera in world.cameras:
                    if camera.name not in seen_cameras:
                        merged.world.cameras.append(camera)
                        seen_cameras.add(camera.name)
                merged.world.terrain = world.terrain
                merged.world.gravity = world.gravity
                merged.table_height = max(merged.table_height, scene.table_height)
                if scene.lighting != "default":
                    merged.lighting = scene.lighting
        return merged

    def _initialize_components(self) -> None:
        scene = self._merged_scene()
        if not isinstance(self._backend, SupportsMultiRobot):
            raise NotImplementedError(
                f"Backend '{type(self._backend).__name__}' does not support multi-robot init. "
                "Implement initialize_multi() / step_multi() / reset_multi() / get_obs_multi() "
                "or use a different backend."
            )
        self._backend.initialize_multi(self._robots, scene, self._shared_sensors)

        for robot in self._robots:
            for robot_task in robot.tasks.values():
                binder = getattr(robot_task.task, "_bind_backend", None)
                if callable(binder):
                    binder(self._backend)

        for sensor in self._all_sensors():
            sensor.initialize(self._backend)
            sensor.warmup()

        try:
            obs_by_robot = self.get_processed_obs_by_robot()
        except Exception:
            obs_by_robot = {}
        if obs_by_robot:
            for robot in self._robots:
                obs = obs_by_robot.get(robot.robot_id)
                if obs is None:
                    continue
                robot.warmup(obs)

        self._bind_policy_runtime()
        self._initialized = True

    def _bind_policy_runtime(self) -> None:
        for robot in self._robots:
            for robot_task in robot.tasks.values():
                for policy in robot_task.policies.values():
                    bind = getattr(policy, "bind_runtime", None)
                    if callable(bind):
                        bind(self._backend, robot.description)

    def _negotiate_action_spaces(self) -> None:
        from robodeploy.policies.learned.diagnostics import PolicyDiagnostics
        from robodeploy.policies.learned.negotiation import negotiate_action_space

        for robot in self._robots:
            for robot_task in robot.tasks.values():
                for policy in robot_task.policies.values():
                    _, effective_space, adapter = negotiate_action_space(
                        policy,
                        self._backend,
                        robot.description,
                        existing_adapter=robot.action_adapter,
                    )
                    robot.action_adapter = adapter
                    robot.effective_action_space = effective_space
            self._policy_diagnostics[robot.robot_id] = PolicyDiagnostics(
                expected_dim=int(getattr(robot.description, "dof", 0) or 0) or None
            )

    def _all_sensors(self) -> list[ISensor]:
        sensors: list[ISensor] = []
        seen: set[int] = set()
        for robot in self._robots:
            for sensor in robot.sensors:
                if id(sensor) not in seen:
                    sensors.append(sensor)
                    seen.add(id(sensor))
        for sensor in self._shared_sensors:
            if id(sensor) not in seen:
                sensors.append(sensor)
                seen.add(id(sensor))
        return sensors

    def get_processed_obs_by_robot(self) -> dict[str, Observation]:
        raw_obs_list = self._backend.get_obs_multi()
        self._require_obs_count(raw_obs_list, "get_obs_multi")
        pending = self._drain_backend_sensor_reads()
        return {
            robot.robot_id: self._process_robot_obs(
                robot, raw_obs_list[idx], pending_reads=pending
            )
            for idx, robot in enumerate(self._robots[: len(raw_obs_list)])
        }

    def _drain_backend_sensor_reads(self) -> list:
        drain = getattr(self._backend, "drain_sensor_reads", None)
        if callable(drain):
            return list(drain())
        return []

    def _process_robot_obs(self, robot: Robot, obs: Observation, *, pending_reads: list) -> Observation:
        for name, sensor_data in pending_reads:
            robot.obs_pipeline.buffer_sensor(name, sensor_data)
        processed = robot.obs_pipeline.process(obs)
        self._validate_obs_against_task(robot, processed)
        return processed

    def _validate_obs_against_task(self, robot: Robot, obs: Observation) -> None:
        if self._obs_spec_policy.lower() == "off":
            return
        task_id = robot.active_task_id
        if not task_id:
            return
        robot_task = robot.tasks.get(task_id)
        if robot_task is None:
            return
        from robodeploy.core.types import validate_observation

        validate_observation(
            obs,
            robot_task.task.obs_spec(),
            policy=self._obs_spec_policy,
        )

    def demo_session(self):
        """Return a DemoSession wrapper that records explicit env.step actions."""
        from robodeploy.demo_recording import DemoSession

        return DemoSession(self)

    def run_episode(
        self,
        steps: int,
        *,
        action_fn=None,
        record: bool = True,
        seed: int | None = None,
    ):
        """Run reset + N steps; optionally record explicit actions into a DemoRecorder."""
        from robodeploy.demo_recording import DemoRecorder, DemoSession

        recorder = DemoRecorder()
        if seed is not None:
            recorder.metadata["seed"] = int(seed)
        session = DemoSession(self, recorder=recorder) if record else None
        obs, info = (session.reset(seed=seed) if session else self.reset(seed=seed))
        for _ in range(int(steps)):
            action = action_fn(obs) if action_fn is not None else None
            if session is not None:
                obs, _, done, info = session.step(action)
            else:
                obs, _, done, info = self.step(action)
            if done:
                break
        if record:
            return recorder
        return obs, info

    def reset(self, *, seed: int | None = None) -> tuple[Observation, EpisodeInfo]:
        if seed is not None:
            self._master_seed = int(seed)
            self._seeds = derive_seeds(self._master_seed)
            self._seed_components()

        if not self._initialized:
            self._initialize_components()
            if self._seeds is not None:
                self._seed_components()

        if self._record_manifest and self._manifest_recorder is None:
            from robodeploy.observability.manifest import ManifestRecorder

            self._manifest_recorder = ManifestRecorder(self, run_name=self._run_name)

        raw_obs_list = self._backend.reset_multi()
        self._require_obs_count(raw_obs_list, "reset_multi")

        policy_seed = self._seeds.policy_seed if self._seeds is not None else None
        for robot in self._robots:
            robot.reset(policy_seed=policy_seed)
            robot.obs_pipeline.reset_sync()
        self._warm_start_action_adapters(raw_obs_list)
        for robot in self._robots:
            for robot_task in robot.tasks.values():
                self._run_task_reset_routine(robot, robot_task)

        pending = self._drain_backend_sensor_reads()
        obs_by_robot = {
            robot.robot_id: self._process_robot_obs(
                robot, raw_obs_list[idx], pending_reads=pending
            )
            for idx, robot in enumerate(self._robots[: len(raw_obs_list)])
        }
        self._episode_info = EpisodeInfo(episode_id=self._episode_info.episode_id + 1)

        primary_robot = self._robots[0]
        primary_obs = obs_by_robot[primary_robot.robot_id]

        info = self._episode_info
        info.extra["multi_agent"] = build_multi_agent_extra(
            self._build_multi_info(obs_by_robot, {}, {}, [])
        )
        info.extra["viz"] = self._build_viz_payload(obs_by_robot)
        self._maybe_send_viz_to_backend(info.extra["viz"])
        info.extra["assets"] = build_assets_extra(getattr(self._backend, "asset_selections", {}))
        info.extra["diagnostics"] = build_diagnostics_extra(self._backend_diagnostics())
        primary_task_id = primary_robot.active_task_id
        primary_task = primary_robot.tasks.get(primary_task_id) if primary_task_id else None
        self._attach_step_observability(
            info,
            obs=primary_obs,
            reward=0.0,
            done=False,
            robot=primary_robot,
            robot_task=primary_task,
            action=Action(),
        )
        self._last_obs_by_robot = dict(obs_by_robot)
        info.extra["safety"] = self._safety_payload()
        return primary_obs, info

    def _run_task_reset_routine(self, robot: Robot, robot_task: RobotTask) -> None:
        if self._safety is not None and self._safety.tripped:
            return
        try:
            for reset_action in robot_task.task.reset_routine(self._backend):
                adapted = robot.action_adapter.process(reset_action)
                action_space = self._robot_action_space(robot)
                safe = robot.description.get_safety_filter().filter(adapted, action_space)
                if self._safety is not None:
                    obs = self._last_obs_by_robot.get(robot.robot_id)
                    if obs is None:
                        obs = make_obs_fallback(robot)
                    safe = self._safety.check_action(
                        safe,
                        obs,
                        dt=self._control_dt(),
                        robot_id=robot.robot_id,
                    )
                self._backend.step_multi(
                    self._hold_actions_for_step_multi(active={robot.robot_id: safe})
                )
        except HumanInterventionRequired as e:
            if self._on_pause:
                self._on_pause()
            print(f"\n[RoboEnv] Human intervention required: {e}")
            if self._on_intervention is not None:
                self._on_intervention(e)
            else:
                input("[RoboEnv] Press Enter when ready to continue...")
            if self._on_resume:
                self._on_resume()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: Optional[Action | List[Action] | Dict[str, Action]] = None):
        try:
            return self._step_impl(action)
        except SafetyError as err:
            self._enter_safe_state()
            primary_robot = self._robots[0]
            primary_obs = self._last_obs_by_robot.get(
                primary_robot.robot_id,
                make_obs_fallback(primary_robot),
            )
            info = self._build_safety_info(err)
            self._episode_info = info
            return primary_obs, 0.0, True, info

    def _step_impl(self, action: Optional[Action | List[Action] | Dict[str, Action]] = None):
        explicit_actions = self._normalize_explicit_actions(action)
        obs_by_robot = self.get_processed_obs_by_robot()
        self._last_obs_by_robot = dict(obs_by_robot)

        final_actions: dict[str, Action] = {}
        prepared_active_ids: dict[str, list[str]] = {}
        precomputed_task_actions: dict[str, dict[str, Action]] = {}
        if explicit_actions is None:
            prepared_active_ids, precomputed_task_actions = self._plan_policy_actions(obs_by_robot)
        dt = self._control_dt()
        for robot in self._robots:
            obs = obs_by_robot[robot.robot_id]
            if explicit_actions is not None and robot.robot_id in explicit_actions:
                safe_action = robot.action_adapter.process(explicit_actions[robot.robot_id])
            else:
                safe_action = robot.step(
                    obs,
                    active_ids=prepared_active_ids.get(robot.robot_id),
                    precomputed_task_actions=precomputed_task_actions.get(robot.robot_id),
                )
            if self._safety is not None:
                safe_action = self._safety.check_action(
                    safe_action,
                    obs,
                    dt=dt,
                    robot_id=robot.robot_id,
                    ignore_slew=(
                        explicit_actions is not None
                        and robot.robot_id in explicit_actions
                    ),
                )
            for task_id in robot.active_task_ids():
                robot_task = robot.tasks.get(task_id)
                if robot_task is not None:
                    safe_action = robot_task.task.transform_action(safe_action)
            final_actions[robot.robot_id] = safe_action

        ordered_actions = [
            final_actions.get(
                robot.robot_id,
                Action(joint_positions=robot.description.home_qpos),
            )
            for robot in self._robots
        ]
        raw_obs_list = self._backend.step_multi(ordered_actions)
        for robot, task_id, robot_task in self._active_task_refs():
            robot_task.task.apply_disturbance(self._backend)
        self._require_obs_count(raw_obs_list, "step_multi")
        pending = self._drain_backend_sensor_reads()
        next_obs_by_robot = {
            robot.robot_id: self._process_robot_obs(
                robot, raw_obs_list[idx], pending_reads=pending
            )
            for idx, robot in enumerate(self._robots[: len(raw_obs_list)])
        }
        self._last_obs_by_robot = dict(next_obs_by_robot)
        if self._safety is not None:
            for robot in self._robots:
                self._safety.check_observation(
                    next_obs_by_robot[robot.robot_id],
                    robot_id=robot.robot_id,
                )

        (
            task_states,
            primary_obs,
            primary_reward,
            primary_done,
            primary_success,
            primary_failure,
            primary_truncated,
            primary_reward_components,
        ) = self._evaluate_active_tasks(next_obs_by_robot, final_actions)

        info = EpisodeInfo(
            episode_id=self._episode_info.episode_id,
            step=self._primary_step(task_states),
            reward=primary_reward,
            success=primary_success,
            failure=primary_failure or (self._safety.tripped if self._safety else False),
        )
        if primary_truncated:
            info.extra["truncated"] = True
            info.extra["timeout"] = True
        info.extra["multi_agent"] = build_multi_agent_extra(
            self._build_multi_info(next_obs_by_robot, final_actions, task_states, [])
        )
        info.extra["viz"] = self._build_viz_payload(next_obs_by_robot)
        self._maybe_send_viz_to_backend(info.extra["viz"])
        info.extra["diagnostics"] = build_diagnostics_extra(self._backend_diagnostics())
        info.extra["safety"] = self._safety_payload()
        if primary_reward_components:
            info.extra["reward_components"] = primary_reward_components
        primary_robot = self._robots[0]
        primary_task_id = primary_robot.active_task_id
        primary_task = primary_robot.tasks.get(primary_task_id) if primary_task_id else None
        primary_action = final_actions.get(primary_robot.robot_id, Action())
        self._attach_step_observability(
            info,
            obs=primary_obs,
            reward=primary_reward,
            done=primary_done,
            robot=primary_robot,
            robot_task=primary_task,
            action=primary_action,
        )
        self._episode_info = info
        return primary_obs, primary_reward, primary_done, info

    def _normalize_explicit_actions(
        self,
        action: Optional[Action | List[Action] | Dict[str, Action]],
    ) -> Optional[dict[str, Action]]:
        if action is None:
            return None
        if isinstance(action, dict):
            unknown = set(action) - {robot.robot_id for robot in self._robots}
            if unknown:
                raise ValueError(f"Explicit action dict contains unknown robot IDs: {sorted(unknown)}")
            return dict(action)
        if isinstance(action, list):
            if len(action) != len(self._robots):
                raise ValueError(
                    f"Explicit action list length mismatch: got {len(action)}, expected {len(self._robots)}."
                )
            return {
                robot.robot_id: action[idx]
                for idx, robot in enumerate(self._robots[: len(action)])
            }
        if isinstance(action, Action):
            return {self._robots[0].robot_id: action}
        raise TypeError("RoboEnv.step() expects Action, list[Action], dict[str, Action], or None.")

    def _require_obs_count(self, raw_obs_list: list[Observation], method: str) -> None:
        if len(raw_obs_list) < len(self._robots):
            raise RuntimeError(
                f"Backend {type(self._backend).__name__}.{method} returned {len(raw_obs_list)} "
                f"observations for {len(self._robots)} robots."
            )

    def _plan_policy_actions(
        self,
        obs_by_robot: dict[str, Observation],
    ) -> tuple[dict[str, list[str]], dict[str, dict[str, Action]]]:
        prepared_active_ids: dict[str, list[str]] = {}
        batch_groups: dict[int, dict[str, object]] = {}

        for robot in self._robots:
            active_ids = robot.prepare_step(obs_by_robot[robot.robot_id])
            prepared_active_ids[robot.robot_id] = active_ids
            for task_id in active_ids:
                robot_task = robot.tasks[task_id]
                if not robot_task.supports_batched_inference():
                    continue
                policy = robot_task.single_policy()
                policy_obs = robot_task.policy_observation(obs_by_robot[robot.robot_id])
                group = batch_groups.setdefault(
                    id(policy),
                    {"policy": policy, "items": []},
                )
                group["items"].append((robot.robot_id, task_id, policy_obs))

        precomputed_actions: dict[str, dict[str, Action]] = {}
        for group in batch_groups.values():
            policy = group["policy"]
            items = list(group["items"])
            if len(items) == 1:
                actions = [policy.get_action(items[0][2])]
            else:
                actions = list(policy.get_action_batch([obs for _, _, obs in items]))
                if len(actions) != len(items):
                    raise RuntimeError(
                        f"Policy {type(policy).__name__}.get_action_batch returned {len(actions)} "
                        f"actions for {len(items)} observations."
                    )
            for (robot_id, task_id, _), planned in zip(items, actions):
                precomputed_actions.setdefault(robot_id, {})[task_id] = planned
        return prepared_active_ids, precomputed_actions

    # ------------------------------------------------------------------
    # Evaluation / metadata
    # ------------------------------------------------------------------

    def _active_task_refs(self) -> list[tuple[Robot, str, RobotTask]]:
        refs: list[tuple[Robot, str, RobotTask]] = []
        for robot in self._robots:
            for task_id in robot.active_task_ids():
                refs.append((robot, task_id, robot.tasks[task_id]))
        return refs

    def _evaluate_active_tasks(
        self,
        obs_by_robot: dict[str, Observation],
        actions_by_robot: dict[str, Action],
    ) -> tuple[
        dict[str, TaskStepState],
        Observation,
        float,
        bool,
        bool,
        bool,
        bool,
        dict[str, float],
    ]:
        task_states: dict[str, TaskStepState] = {}
        primary_robot = self._robots[0]
        primary_obs = obs_by_robot[primary_robot.robot_id]
        primary_task_id = primary_robot.active_task_id
        primary_reward = 0.0
        primary_done = False
        primary_success = False
        primary_failure = False
        primary_truncated = False
        primary_reward_components: dict[str, float] = {}

        for robot, task_id, robot_task in self._active_task_refs():
            robot_task.task._on_step()
            task_obs = obs_by_robot[robot.robot_id]
            action_used = actions_by_robot.get(robot.robot_id, Action())
            reward = robot_task.task.reward_fn(task_obs, action_used)
            success = robot_task.task.success_fn(task_obs)
            task_failure = bool(robot_task.task.failure_fn(task_obs))
            step_budget = min(int(self._max_steps), int(robot_task.task.max_steps()))
            truncated = (
                not success
                and not task_failure
                and robot_task.task.step_count >= step_budget
            )
            failure = task_failure
            done = success or failure or truncated
            unique_id = self._unique_task_state_key(robot.robot_id, task_id)
            task_states[unique_id] = TaskStepState(
                task_id=task_id,
                robot_ids=[robot.robot_id],
                reward=reward,
                done=done,
                success=success,
                failure=failure,
                step=robot_task.task.step_count,
                active=True,
            )
            if robot.robot_id == primary_robot.robot_id and task_id == primary_task_id:
                primary_obs = task_obs
                primary_reward = reward
                primary_done = done
                primary_success = success
                primary_failure = failure
                primary_truncated = truncated
                components_fn = getattr(robot_task.task, "reward_components_fn", None)
                if components_fn is not None:
                    primary_reward_components = components_fn(task_obs, action_used)

        return (
            task_states,
            primary_obs,
            primary_reward,
            primary_done,
            primary_success,
            primary_failure,
            primary_truncated,
            primary_reward_components,
        )

    @staticmethod
    def _unique_task_state_key(robot_id: str, task_id: str) -> str:
        return f"{robot_id}::{task_id}"

    def _primary_step(self, task_states: dict[str, TaskStepState]) -> int:
        primary_robot = self._robots[0]
        key = self._unique_task_state_key(primary_robot.robot_id, primary_robot.active_task_id or "")
        state = task_states.get(key)
        return state.step if state else 0

    def _build_multi_info(
        self,
        obs_by_robot: dict[str, Observation],
        actions_by_robot: dict[str, Action],
        task_states: dict[str, TaskStepState],
        rejected: list[dict],
    ) -> MultiAgentInfo:
        primary_robot = self._robots[0]
        multi_info = MultiAgentInfo(primary_task_id=primary_robot.active_task_id or "")

        events = []
        for robot in self._robots:
            events.extend(robot.consume_arbitration_events())
            multi_info.active_tasks_by_robot[robot.robot_id] = robot.active_task_id or ""
            multi_info.robot_states[robot.robot_id] = RobotStepState(
                robot_id=robot.robot_id,
                active_task_id=robot.active_task_id or "",
                action=actions_by_robot.get(robot.robot_id),
                obs=obs_by_robot.get(robot.robot_id),
            )
        multi_info.arbitration_events = events

        if task_states:
            multi_info.task_states.update(task_states)
        else:
            for robot in self._robots:
                for task_id, robot_task in robot.tasks.items():
                    key = self._unique_task_state_key(robot.robot_id, task_id)
                    multi_info.task_states[key] = TaskStepState(
                        task_id=task_id,
                        robot_ids=[robot.robot_id],
                        active=task_id in robot.active_task_ids(),
                    )

        multi_info.rejected_actions = rejected
        return multi_info

    def _build_viz_payload(self, obs_by_robot: dict[str, Observation]) -> dict:
        tasks: dict[str, dict] = {}
        for robot, task_id, robot_task in self._active_task_refs():
            obs = obs_by_robot.get(robot.robot_id)
            try:
                items = list(robot_task.task.viz_goals(obs))
            except Exception:
                items = []
            entry = tasks.setdefault(task_id, {"per_robot": {}})
            entry["per_robot"][robot.robot_id] = items
        return build_viz_extra(tasks)

    def _maybe_send_viz_to_backend(self, payload: dict) -> None:
        setter = getattr(self._backend, "set_viz_payload", None)
        if isinstance(self._backend, SupportsVizSink) or callable(setter):
            try:
                setter(payload)
            except Exception:
                return

    def _backend_diagnostics(self) -> dict:
        from robodeploy.builtins import failed_builtin_imports

        payload: dict = {"failed_builtin_imports": failed_builtin_imports()}
        getter = getattr(self._backend, "get_diagnostics", None)
        if isinstance(self._backend, SupportsDiagnostics) or callable(getter):
            try:
                payload.update(getter())
            except Exception:
                pass
        return payload

    def _attach_step_observability(
        self,
        info: EpisodeInfo,
        *,
        obs: Observation,
        reward: float,
        done: bool,
        robot: Robot,
        robot_task: RobotTask | None,
        action: Action,
    ) -> None:
        """Surface sensor health and optional structured logging each step."""
        status = dict(getattr(obs, "sensor_status", {}) or {})
        health_summary = summarize_sensor_health(status)
        info.extra["sensor_status"] = status
        info.extra["sensor_health"] = health_summary
        info.extra["backend_diagnostics"] = self._backend_diagnostics()
        info.extra["health_status"] = self._health_monitor.observe(status)
        diag = self._policy_diagnostics.get(robot.robot_id)
        if diag is not None:
            diag.record(action)
            info.extra["policy_diagnostics"] = diag.summary()
        if self._logger is not None:
            self._logger.log_step(
                {
                    "episode_id": info.episode_id,
                    "reward": reward,
                    "done": done,
                    "sensor_health": health_summary.get("overall", "ok"),
                    "sensor_status": status,
                    "robot_id": robot.robot_id,
                    "task_id": robot.active_task_id,
                    "action": action,
                },
                step=info.step,
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self, *, manifest_dir: str | None = None) -> None:
        from robodeploy.safety.registry import clear_safety_monitor

        clear_safety_monitor(self._safety)
        if self._manifest_recorder is not None and manifest_dir:
            self._manifest_recorder.write(manifest_dir)
        if self._logger is not None:
            self._logger.close()
        for robot in self._robots:
            for sensor in robot.sensors:
                sensor.close()
        for sensor in self._shared_sensors:
            sensor.close()
        self._backend.close()
        self._initialized = False

    def render(self) -> None:
        self._backend.render()

    def __repr__(self) -> str:
        return f"RoboEnv(robots={len(self._robots)}, backend={self._backend!r}, real={self.is_real})"


def make_obs_fallback(robot: Robot) -> Observation:
    """Minimal observation when safety halts before any obs is cached."""
    try:
        import jax.numpy as jnp
    except Exception:
        import numpy as jnp  # type: ignore[assignment]

    home = jnp.asarray(robot.description.home_qpos, dtype=jnp.float32)
    dof = int(home.shape[0])
    zeros = jnp.zeros((dof,), dtype=jnp.float32)
    return Observation(
        joint_positions=home,
        joint_velocities=zeros,
        joint_torques=zeros,
        ee_position=jnp.zeros((3,), dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
    )
