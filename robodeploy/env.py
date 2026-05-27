"""RoboEnv — robot-centric orchestrator.

Constructed with a backend and a list of `Robot` aggregates. Each `Robot`
encapsulates its own description, sensors, tasks, policies, and arbitration
choice. RoboEnv steps every robot, evaluates rewards/success per active task,
and merges the per-robot observations / states into a single EpisodeInfo.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from robodeploy.backends.capabilities import SupportsDiagnostics, SupportsMultiRobot, SupportsVizSink
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
    PropConfig,
    RobotStepState,
    SceneSpec,
    TaskStepState,
)
from robodeploy.obs_pipeline import ObsPipeline


class RoboEnv:
    """Robot-centric env. One backend, many robots, one step path."""

    def __init__(
        self,
        backend: IBackend,
        robots: List[Robot],
        *,
        shared_sensors: Optional[List[ISensor]] = None,
        max_episode_steps: Optional[int] = None,
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
        self._episode_info = EpisodeInfo()
        self._initialized = False
        self._on_pause: Optional[Callable[[], None]] = None
        self._on_resume: Optional[Callable[[], None]] = None
        self._on_intervention: Optional[Callable[[HumanInterventionRequired], None]] = None

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
        backend_kwargs: Optional[dict] = None,
        task_kwargs: Optional[dict] = None,
        policy_kwargs: Optional[dict] = None,
        obs_pipeline: Optional[ObsPipeline] = None,
        robot_id: str = "robot0",
        task_id: str = "task0",
        policy_id: str = "policy0",
    ) -> "RoboEnv":
        from robodeploy.builtins import import_builtins

        import_builtins()
        DescriptionClass = get_robot(robot)
        BackendClass = get_backend(backend)
        TaskClass = get_task(task)

        description_obj = DescriptionClass()
        backend_obj = BackendClass(**(backend_kwargs or {}))
        task_obj: ITask = TaskClass(**(task_kwargs or {}))
        sensor_backend_name = cls._sensor_backend_name_for(backend_obj, default_name=backend)

        if policy is None:
            raise ValueError(
                "RoboEnv.make() requires a policy name. "
                "Construct RoboEnv(robots=[...]) directly to use external action injection."
            )
        PolicyClass = get_policy(policy)
        policy_obj: IPolicy = PolicyClass(**(policy_kwargs or {}))

        sensor_objs: List[ISensor] = []
        if sensors:
            for s in sensors:
                SensorClass = resolve_sensor_class(
                    s,
                    is_real=backend_obj.is_real,
                    backend_name=sensor_backend_name,
                )
                cfg = dict((sensor_kwargs or {}).get(s, {}) or {})
                try:
                    sensor_objs.append(SensorClass(config=cfg))
                except TypeError:
                    sensor_objs.append(SensorClass())

        robot_obj = Robot(
            robot_id=robot_id,
            description=description_obj,
            tasks={
                task_id: RobotTask(
                    task=task_obj,
                    policies={policy_id: policy_obj},
                    task_id=task_id,
                ),
            },
            sensors=sensor_objs,
            obs_pipeline=obs_pipeline or ObsPipeline(),
        )

        return cls(backend=backend_obj, robots=[robot_obj])

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        obs_pipeline: Optional[ObsPipeline] = None,
    ) -> "RoboEnv":
        from robodeploy.builtins import import_builtins
        from robodeploy.core.registry import use

        import_builtins()
        cfg = dict(cfg)
        for module_path in cfg.pop("custom_modules", []):
            use(module_path)

        backend_obj = cls._coerce_backend(cfg["backend"], cfg.get("backend_kwargs"))
        sensor_backend_name = cls._sensor_backend_name_for(
            backend_obj,
            default_name=cfg["backend"] if isinstance(cfg.get("backend"), str) else None,
        )
        if "robots" in cfg:
            robots = [cls._coerce_robot_object(item) for item in cfg["robots"]]
            return cls(
                backend=backend_obj,
                robots=robots,
                shared_sensors=cfg.get("shared_sensors"),
                max_episode_steps=cfg.get("max_episode_steps"),
            )

        robot_value = cfg["robot"]
        if isinstance(robot_value, Robot):
            return cls(
                backend=backend_obj,
                robots=[robot_value],
                shared_sensors=cfg.get("shared_sensors"),
                max_episode_steps=cfg.get("max_episode_steps"),
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
            is_real=backend_obj.is_real,
            backend_name=sensor_backend_name,
        )

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
            obs_pipeline=obs_pipeline or ObsPipeline(),
        )
        return cls(
            backend=backend_obj,
            robots=[robot_obj],
            shared_sensors=cfg.get("shared_sensors"),
            max_episode_steps=cfg.get("max_episode_steps"),
        )

    @staticmethod
    def _instantiate_component(value: Any, kwargs: Optional[dict]) -> Any:
        if isinstance(value, type):
            return value(**(kwargs or {}))
        if callable(value) and not isinstance(value, (str, Robot)):
            return value(**(kwargs or {}))
        return value

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
            return TaskClass(**(kwargs or {}))
        obj = cls._instantiate_component(value, kwargs)
        if not isinstance(obj, ITask):
            raise TypeError("task must be a registry name, task class, or ITask instance.")
        return obj

    @classmethod
    def _coerce_policy(cls, value: Any, kwargs: Optional[dict]) -> IPolicy:
        if isinstance(value, str):
            PolicyClass = get_policy(value)
            return PolicyClass(**(kwargs or {}))
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

        self._initialized = True

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
        return robot.obs_pipeline.process(obs)

    def demo_session(self):
        """Return a DemoSession wrapper that records explicit env.step actions."""
        from robodeploy.demo_recording import DemoSession

        return DemoSession(self)

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "RoboEnv":
        """Build RoboEnv via RoboEnv.make using a named YAML preset."""
        from robodeploy.builtins import import_builtins
        from robodeploy.config import load_preset

        import_builtins()
        cfg = {**load_preset(preset_name), **overrides}
        return cls.make(
            robot=str(cfg["robot"]),
            backend=str(cfg["backend"]),
            task=str(cfg["task"]),
            policy=str(cfg["policy"]),
            robot_id=str(cfg.get("robot_id", "robot0")),
            task_id=str(cfg.get("task_id", "task0")),
            policy_id=str(cfg.get("policy_id", "policy0")),
            backend_kwargs=cfg.get("backend_kwargs"),
            task_kwargs=cfg.get("task_kwargs"),
            policy_kwargs=cfg.get("policy_kwargs"),
            sensor_kwargs=cfg.get("sensor_kwargs"),
        )

    def reset(self) -> tuple[Observation, EpisodeInfo]:
        if not self._initialized:
            self._initialize_components()

        raw_obs_list = self._backend.reset_multi()
        self._require_obs_count(raw_obs_list, "reset_multi")

        for robot in self._robots:
            robot.reset()
            robot.obs_pipeline.reset_sync()
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
        return primary_obs, info

    def _run_task_reset_routine(self, robot: Robot, robot_task: RobotTask) -> None:
        try:
            for reset_action in robot_task.task.reset_routine(self._backend):
                adapted = robot.action_adapter.process(reset_action)
                action_space = robot_task.action_space()
                safe = robot.description.get_safety_filter().filter(adapted, action_space)
                self._backend.step_multi([safe])
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
        explicit_actions = self._normalize_explicit_actions(action)
        obs_by_robot = self.get_processed_obs_by_robot()

        final_actions: dict[str, Action] = {}
        prepared_active_ids: dict[str, list[str]] = {}
        precomputed_task_actions: dict[str, dict[str, Action]] = {}
        if explicit_actions is None:
            prepared_active_ids, precomputed_task_actions = self._plan_policy_actions(obs_by_robot)
        for robot in self._robots:
            obs = obs_by_robot[robot.robot_id]
            if explicit_actions is not None and robot.robot_id in explicit_actions:
                supplied = explicit_actions[robot.robot_id]
                adapted = robot.action_adapter.process(supplied)
                first_task = next(iter(robot.tasks.values()))
                action_space = first_task.action_space()
                final_actions[robot.robot_id] = robot.description.get_safety_filter().filter(adapted, action_space)
            else:
                final_actions[robot.robot_id] = robot.step(
                    obs,
                    active_ids=prepared_active_ids.get(robot.robot_id),
                    precomputed_task_actions=precomputed_task_actions.get(robot.robot_id),
                )

        ordered_actions = [
            final_actions.get(
                robot.robot_id,
                Action(joint_positions=robot.description.home_qpos),
            )
            for robot in self._robots
        ]
        raw_obs_list = self._backend.step_multi(ordered_actions)
        self._require_obs_count(raw_obs_list, "step_multi")
        pending = self._drain_backend_sensor_reads()
        next_obs_by_robot = {
            robot.robot_id: self._process_robot_obs(
                robot, raw_obs_list[idx], pending_reads=pending
            )
            for idx, robot in enumerate(self._robots[: len(raw_obs_list)])
        }

        task_states, primary_obs, primary_reward, primary_done, primary_success, primary_failure = (
            self._evaluate_active_tasks(next_obs_by_robot, final_actions)
        )

        info = EpisodeInfo(
            episode_id=self._episode_info.episode_id,
            step=self._primary_step(task_states),
            reward=primary_reward,
            success=primary_success,
            failure=primary_failure,
        )
        info.extra["multi_agent"] = build_multi_agent_extra(
            self._build_multi_info(next_obs_by_robot, final_actions, task_states, [])
        )
        info.extra["viz"] = self._build_viz_payload(next_obs_by_robot)
        self._maybe_send_viz_to_backend(info.extra["viz"])
        info.extra["diagnostics"] = build_diagnostics_extra(self._backend_diagnostics())
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
    ) -> tuple[dict[str, TaskStepState], Observation, float, bool, bool, bool]:
        task_states: dict[str, TaskStepState] = {}
        primary_robot = self._robots[0]
        primary_obs = obs_by_robot[primary_robot.robot_id]
        primary_task_id = primary_robot.active_task_id
        primary_reward = 0.0
        primary_done = False
        primary_success = False
        primary_failure = False

        for robot, task_id, robot_task in self._active_task_refs():
            robot_task.task._on_step()
            task_obs = obs_by_robot[robot.robot_id]
            action_used = actions_by_robot.get(robot.robot_id, Action())
            reward = robot_task.task.reward_fn(task_obs, action_used)
            success = robot_task.task.success_fn(task_obs)
            failure = robot_task.task.failure_fn(task_obs) or (
                robot_task.task.step_count >= robot_task.task.max_steps()
            )
            done = success or failure
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

        return task_states, primary_obs, primary_reward, primary_done, primary_success, primary_failure

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

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
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
