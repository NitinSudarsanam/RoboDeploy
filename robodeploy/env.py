"""RoboEnv — robot-centric orchestrator.

Constructed with a backend and a list of `Robot` aggregates. Each `Robot`
encapsulates its own description, sensors, tasks, policies, and arbitration
choice. RoboEnv steps every robot, evaluates rewards/success per active task,
and merges the per-robot observations / states into a single EpisodeInfo.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

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
from robodeploy.core.registry import get_backend, get_policy, get_robot, get_sensor, get_task
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

        if policy is None:
            raise ValueError(
                "RoboEnv.make() requires a policy name. "
                "Construct RoboEnv(robots=[...]) directly to use external action injection."
            )
        PolicyClass = get_policy(policy)
        policy_obj: IPolicy = PolicyClass(**(policy_kwargs or {}))

        sensor_objs: List[ISensor] = []
        if sensors:
            suffix = "_real" if backend_obj.is_real else "_sim"
            for s in sensors:
                SensorClass = get_sensor(s + suffix)
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
        from robodeploy.core.registry import use

        cfg = dict(cfg)
        for module_path in cfg.pop("custom_modules", []):
            use(module_path)

        return cls.make(
            robot=cfg["robot"],
            backend=cfg["backend"],
            task=cfg["task"],
            policy=cfg.get("policy"),
            sensors=cfg.get("sensors"),
            backend_kwargs=cfg.get("backend_kwargs"),
            task_kwargs=cfg.get("task_kwargs"),
            policy_kwargs=cfg.get("policy_kwargs"),
            obs_pipeline=obs_pipeline,
        )

    # ------------------------------------------------------------------
    # External hooks
    # ------------------------------------------------------------------

    def set_pause_hooks(self, on_pause: Callable[[], None], on_resume: Callable[[], None]) -> None:
        self._on_pause = on_pause
        self._on_resume = on_resume

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
        seen: set[str] = set()
        for robot in self._robots:
            for robot_task in robot.tasks.values():
                scene = robot_task.task.scene_spec()
                for prop in getattr(scene, "props", []) or []:
                    if prop.name not in seen:
                        merged.props.append(prop)
                        seen.add(prop.name)
                for obj in scene.objects or []:
                    if obj.name not in seen:
                        merged.objects.append(obj)
                        seen.add(obj.name)
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
            for sensor in robot.sensors:
                sensor.warmup()
        for sensor in self._shared_sensors:
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

    def get_processed_obs_by_robot(self) -> dict[str, Observation]:
        raw_obs_list = self._backend.get_obs_multi()
        return {
            robot.robot_id: robot.obs_pipeline.process(raw_obs_list[idx])
            for idx, robot in enumerate(self._robots[: len(raw_obs_list)])
        }

    def reset(self) -> tuple[Observation, EpisodeInfo]:
        if not self._initialized:
            self._initialize_components()

        raw_obs_list = self._backend.reset_multi()

        for robot in self._robots:
            robot.reset()
            for robot_task in robot.tasks.values():
                self._run_task_reset_routine(robot, robot_task)

        obs_by_robot = {
            robot.robot_id: robot.obs_pipeline.process(raw_obs_list[idx])
            for idx, robot in enumerate(self._robots[: len(raw_obs_list)])
        }
        self._episode_info = EpisodeInfo(episode_id=self._episode_info.episode_id + 1)

        primary_robot = self._robots[0]
        primary_obs = obs_by_robot.get(primary_robot.robot_id, raw_obs_list[0] if raw_obs_list else Observation(
            joint_positions=primary_robot.description.home_qpos,
            joint_velocities=primary_robot.description.home_qpos * 0.0,
            joint_torques=primary_robot.description.home_qpos * 0.0,
            ee_position=primary_robot.description.home_qpos[:3] * 0.0,
            ee_orientation=primary_robot.description.home_qpos[:4] * 0.0,
            ee_velocity=primary_robot.description.home_qpos[:3] * 0.0,
            ee_angular_velocity=primary_robot.description.home_qpos[:3] * 0.0,
        ))

        info = EpisodeInfo(episode_id=self._episode_info.episode_id)
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
        for robot in self._robots:
            obs = obs_by_robot[robot.robot_id]
            if explicit_actions is not None and robot.robot_id in explicit_actions:
                supplied = explicit_actions[robot.robot_id]
                adapted = robot.action_adapter.process(supplied)
                first_task = next(iter(robot.tasks.values()))
                action_space = first_task.action_space()
                final_actions[robot.robot_id] = robot.description.get_safety_filter().filter(adapted, action_space)
            else:
                final_actions[robot.robot_id] = robot.step(obs)

        ordered_actions = [
            final_actions.get(
                robot.robot_id,
                Action(joint_positions=robot.description.home_qpos),
            )
            for robot in self._robots
        ]
        raw_obs_list = self._backend.step_multi(ordered_actions)
        next_obs_by_robot = {
            robot.robot_id: robot.obs_pipeline.process(raw_obs_list[idx])
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
            return dict(action)
        if isinstance(action, list):
            return {
                robot.robot_id: action[idx]
                for idx, robot in enumerate(self._robots[: len(action)])
            }
        if isinstance(action, Action):
            return {self._robots[0].robot_id: action}
        raise TypeError("RoboEnv.step() expects Action, list[Action], dict[str, Action], or None.")

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
        getter = getattr(self._backend, "get_diagnostics", None)
        if isinstance(self._backend, SupportsDiagnostics) or callable(getter):
            try:
                return getter()
            except Exception:
                return {}
        return {}

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
