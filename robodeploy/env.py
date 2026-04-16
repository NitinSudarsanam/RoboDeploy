"""RoboEnv — single-agent and multi-agent orchestrator."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from robodeploy.action_adapter import ActionAdapter
from robodeploy.core.arbitrator import Arbitrator
from robodeploy.core.interfaces.backend import IBackend
from robodeploy.core.interfaces.policy import IPolicy
from robodeploy.core.interfaces.sensor import ISensor
from robodeploy.core.interfaces.task import ITask
from robodeploy.core.registry import get_backend, get_policy, get_robot, get_sensor, get_task
from robodeploy.core.robot_config import RobotConfig
from robodeploy.core.task_config import TaskConfig
from robodeploy.core.types import (
    Action,
    EpisodeInfo,
    HumanInterventionRequired,
    MultiAgentInfo,
    Observation,
    RobotStepState,
    SceneSpec,
    TaskStepState,
)
from robodeploy.description.base import RobotDescription
from robodeploy.obs_pipeline import ObsPipeline


class RoboEnv:
    """Gym-style env supporting single-agent and full multi-agent routing."""

    def __init__(
        self,
        description: Optional[RobotDescription] = None,
        backend: Optional[IBackend] = None,
        task: Optional[ITask] = None,
        policy: Optional[IPolicy] = None,
        sensors: Optional[list[ISensor]] = None,
        obs_pipeline: Optional[ObsPipeline] = None,
        action_adapter: Optional[ActionAdapter] = None,
        max_episode_steps: Optional[int] = None,
        *,
        robots: Optional[List[RobotConfig]] = None,
        tasks: Optional[List[TaskConfig]] = None,
        shared_sensors: Optional[List[ISensor]] = None,
        action_resolvers: Optional[Dict[str, Callable[[str, list[Action]], Action]]] = None,
    ) -> None:
        if backend is None:
            raise ValueError("RoboEnv requires a backend.")

        self._backend = backend
        self._single_agent_mode = robots is None and tasks is None

        if self._single_agent_mode:
            if description is None or task is None:
                raise ValueError("Single-agent RoboEnv requires description and task.")
            self._robots = [
                RobotConfig(
                    description=description,
                    obs_pipeline=obs_pipeline or ObsPipeline(),
                    action_adapter=action_adapter or ActionAdapter(),
                    sensors=sensors or [],
                    robot_id="robot0",
                )
            ]
            self._tasks_cfg = [
                TaskConfig(
                    task=task,
                    robot_ids=["robot0"],
                    policy=policy,
                    task_id="task0",
                    mode="sequential",
                )
            ]
        else:
            self._robots = robots or []
            self._tasks_cfg = tasks or []
            if not self._robots or not self._tasks_cfg:
                raise ValueError("Multi-agent RoboEnv requires non-empty robots and tasks.")

        for idx, robot in enumerate(self._robots):
            if not robot.robot_id:
                robot.robot_id = f"robot{idx}"
        for idx, task_cfg in enumerate(self._tasks_cfg):
            if not task_cfg.task_id:
                task_cfg.task_id = f"task{idx}"

        self._robot_by_id = {robot.robot_id: robot for robot in self._robots}
        self._shared_sensors = shared_sensors or []
        self._action_resolvers = action_resolvers or {}
        self._tasks_by_id = {task_cfg.task_id: task_cfg for task_cfg in self._tasks_cfg}
        self._arbitrator = Arbitrator(self._tasks_cfg)

        self._description = self._robots[0].description
        self._task = self._tasks_cfg[0].task
        self._policy = self._tasks_cfg[0].policy
        self._sensors = self._robots[0].sensors
        self._pipeline = self._robots[0].obs_pipeline
        self._adapter = self._robots[0].action_adapter
        self._safety = self._description.get_safety_filter()
        self._max_steps = max_episode_steps or self._task.max_steps()
        self._primary_task_id = self._tasks_cfg[0].task_id

        self._episode_info = EpisodeInfo()
        self._initialized = False
        self._on_pause: Optional[Callable] = None
        self._on_resume: Optional[Callable] = None

    def set_pause_hooks(self, on_pause: Callable, on_resume: Callable) -> None:
        self._on_pause = on_pause
        self._on_resume = on_resume

    @classmethod
    def make(
        cls,
        robot: str,
        backend: str,
        task: str,
        policy: Optional[str] = None,
        sensors: Optional[list[str]] = None,
        backend_kwargs: Optional[dict] = None,
        task_kwargs: Optional[dict] = None,
        policy_kwargs: Optional[dict] = None,
        obs_pipeline: Optional[ObsPipeline] = None,
    ) -> "RoboEnv":
        DescriptionClass = get_robot(robot)
        BackendClass = get_backend(backend)
        TaskClass = get_task(task)

        description_obj = DescriptionClass()
        backend_obj = BackendClass(**(backend_kwargs or {}))
        task_obj = TaskClass(**(task_kwargs or {}))

        policy_obj: Optional[IPolicy] = None
        if policy:
            PolicyClass = get_policy(policy)
            policy_obj = PolicyClass(**(policy_kwargs or {}))

        sensor_objs: list[ISensor] = []
        if sensors:
            suffix = "_real" if backend_obj.is_real else "_sim"
            for s in sensors:
                SensorClass = get_sensor(s + suffix)
                sensor_objs.append(SensorClass())

        return cls(
            description=description_obj,
            backend=backend_obj,
            task=task_obj,
            policy=policy_obj,
            sensors=sensor_objs,
            obs_pipeline=obs_pipeline,
        )

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

    def switch_task(self, robot_id: str, to_task_id: str, reason: str = "") -> None:
        self._arbitrator.switch(robot_id, to_task_id, reason=reason)
        self._primary_task_id = self._primary_task().task_id

    def _primary_task(self) -> TaskConfig:
        if self._primary_task_id in self._tasks_by_id:
            return self._tasks_by_id[self._primary_task_id]
        return self._tasks_cfg[0]

    def _active_tasks(self) -> list[TaskConfig]:
        active: list[TaskConfig] = []
        for task_cfg in self._tasks_cfg:
            if task_cfg.mode == "concurrent":
                active.append(task_cfg)
                continue
            if all(self._arbitrator.active_task_id(robot_id) == task_cfg.task_id for robot_id in task_cfg.robot_ids):
                active.append(task_cfg)
        return active

    def _merged_scene_spec(self) -> SceneSpec:
        merged = SceneSpec()
        seen: set[str] = set()
        for task_cfg in self._tasks_cfg:
            scene = task_cfg.task.scene_spec()
            for obj in scene.objects:
                if obj.name not in seen:
                    merged.objects.append(obj)
                    seen.add(obj.name)
            merged.table_height = max(merged.table_height, scene.table_height)
            if scene.lighting != "default":
                merged.lighting = scene.lighting
        return merged

    def _initialize_components(self) -> None:
        if self._single_agent_mode:
            self._backend.initialize(self._description, self._task, self._sensors)
        else:
            self._backend.initialize_multi(self._robots, self._merged_scene_spec(), self._shared_sensors)

        for robot in self._robots:
            for sensor in robot.sensors:
                sensor.warmup()
        for sensor in self._shared_sensors:
            sensor.warmup()

        try:
            obs_by_robot = self.get_processed_obs_by_robot()
        except Exception:
            obs_by_robot = {}

        for task_cfg in self._tasks_cfg:
            if task_cfg.policy is None or not obs_by_robot:
                continue
            try:
                sample_obs = obs_by_robot[task_cfg.robot_ids[0]]
                task_cfg.policy.warmup(sample_obs)
            except Exception as exc:
                print(f"[RoboEnv] Policy warmup failed (non-fatal): {exc}")

        self._initialized = True

    def get_processed_obs_by_robot(self) -> dict[str, Observation]:
        raw_obs_list = self._backend.get_obs_multi() if not self._single_agent_mode else [self._backend.get_obs()]
        return {
            robot.robot_id: robot.obs_pipeline.process(raw_obs_list[idx])
            for idx, robot in enumerate(self._robots[: len(raw_obs_list)])
        }

    def _run_task_reset_routine(self, task_cfg: TaskConfig) -> None:
        if len(task_cfg.robot_ids) != 1:
            return
        rid = task_cfg.robot_ids[0]
        robot = self._robot_by_id[rid]
        robot.action_adapter.reset()
        try:
            for reset_action in task_cfg.task.reset_routine(self._backend):
                adapted = robot.action_adapter.process(reset_action)
                safe = robot.description.get_safety_filter().filter(
                    adapted,
                    self._infer_action_space(adapted),
                )
                self._backend.step(safe)
        except HumanInterventionRequired as e:
            if self._on_pause:
                self._on_pause()
            print(f"\n[RoboEnv] Human intervention required: {e}")
            input("[RoboEnv] Press Enter when ready to continue...")
            if self._on_resume:
                self._on_resume()

    def reset(self) -> tuple[Observation, EpisodeInfo]:
        if not self._initialized:
            self._initialize_components()

        if self._single_agent_mode:
            raw_obs_list = [self._backend.reset()]
        else:
            raw_obs_list = self._backend.reset_multi()

        for task_cfg in self._tasks_cfg:
            task_cfg.task._on_reset()
            if task_cfg.policy is not None:
                task_cfg.policy.reset()
                task_cfg.policy.set_instruction(task_cfg.task.language_instruction())
            self._run_task_reset_routine(task_cfg)

        obs_by_robot = {
            robot.robot_id: robot.obs_pipeline.process(raw_obs_list[idx])
            for idx, robot in enumerate(self._robots[: len(raw_obs_list)])
        }
        self._episode_info = EpisodeInfo(episode_id=self._episode_info.episode_id + 1)
        return self._make_primary_reset_return(obs_by_robot)

    def _make_primary_reset_return(self, obs_by_robot: dict[str, Observation]) -> tuple[Observation, EpisodeInfo]:
        primary_task = self._primary_task()
        primary_robot_id = primary_task.robot_ids[0]
        info = EpisodeInfo(episode_id=self._episode_info.episode_id)
        info.extra["multi_agent"] = self._build_multi_info(obs_by_robot, {}, {}, {})
        return obs_by_robot[primary_robot_id], info

    def step(self, action: Optional[Action | List[Action] | Dict[str, Action]] = None):
        if self._single_agent_mode:
            return self._step_single(action if isinstance(action, Action) or action is None else list(action.values())[0] if isinstance(action, dict) else action[0])
        return self._step_multi(action)

    def _step_single(self, action: Optional[Action]):
        if action is None:
            if self._policy is None:
                raise RuntimeError(
                    "No action provided and no policy set. "
                    "Pass an action to step() or provide a policy at construction."
                )
            raw_obs = self._backend.get_obs()
            obs = self._pipeline.process(raw_obs)
            action = self._policy.get_action(obs)

        action = self._adapter.process(action)
        safe_action = self._safety.filter(
            action,
            self._policy.action_space if self._policy else self._infer_action_space(action),
        )
        raw_obs = self._backend.step(safe_action)
        self._task._on_step()
        obs = self._pipeline.process(raw_obs)
        reward = self._task.reward_fn(obs, safe_action)
        success = self._task.success_fn(obs)
        failure = self._task.failure_fn(obs) or (self._task.step_count >= self._max_steps)
        done = success or failure
        self._episode_info = EpisodeInfo(
            episode_id=self._episode_info.episode_id,
            step=self._task.step_count,
            reward=reward,
            success=success,
            failure=failure,
        )
        return obs, reward, done, self._episode_info

    def _normalize_explicit_actions(
        self,
        action: Optional[List[Action] | Dict[str, Action] | Action],
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
        raise TypeError("Multi-agent RoboEnv.step() expects list[Action], dict[str, Action], or None.")

    def _resolve_task_candidates(
        self,
        obs_by_robot: dict[str, Observation],
        explicit_actions: Optional[dict[str, Action]] = None,
    ) -> dict[str, list[tuple[TaskConfig, Action]]]:
        candidate_actions: dict[str, list[tuple[TaskConfig, Action]]] = {
            robot.robot_id: [] for robot in self._robots
        }
        active_tasks = self._active_tasks()

        if explicit_actions is not None:
            primary_cfg = self._primary_task()
            for robot_id, supplied_action in explicit_actions.items():
                if robot_id in candidate_actions:
                    candidate_actions[robot_id].append((primary_cfg, supplied_action))
            return candidate_actions

        for task_cfg in active_tasks:
            if task_cfg.policy is None:
                raise RuntimeError(
                    f"TaskConfig '{task_cfg.task_id}' has no policy. "
                    "Provide explicit actions or attach a policy."
                )
            task_obs = [obs_by_robot[robot_id] for robot_id in task_cfg.robot_ids]
            produced = (
                [task_cfg.policy.get_action(task_obs[0])]
                if len(task_obs) == 1
                else task_cfg.policy.get_action_batch(task_obs)
            )
            for robot_id, raw_action in zip(task_cfg.robot_ids, produced):
                candidate_actions[robot_id].append((task_cfg, raw_action))
        return candidate_actions

    def _resolve_robot_actions(
        self,
        candidate_actions: dict[str, list[tuple[TaskConfig, Action]]],
        obs_by_robot: dict[str, Observation],
    ) -> tuple[dict[str, Action], list[dict]]:
        final_actions: dict[str, Action] = {}
        rejected: list[dict] = []

        for robot_id, candidates in candidate_actions.items():
            if not candidates:
                continue
            if len(candidates) == 1:
                task_cfg, action = candidates[0]
            else:
                resolver_names = {candidate.action_resolver for candidate, _ in candidates if candidate.action_resolver}
                if len(resolver_names) != 1:
                    raise RuntimeError(
                        f"Robot '{robot_id}' received multiple actions without a shared resolver."
                    )
                resolver_name = next(iter(resolver_names))
                resolver = self._action_resolvers.get(resolver_name)
                if resolver is None:
                    raise RuntimeError(
                        f"Resolver '{resolver_name}' for robot '{robot_id}' is not registered."
                    )
                task_cfg, _ = candidates[-1]
                action = resolver(robot_id, [candidate_action for _, candidate_action in candidates])
                for rejected_cfg, rejected_action in candidates[:-1]:
                    if rejected_cfg.policy is not None:
                        rejected_cfg.policy.notify_rejected(obs_by_robot[robot_id], rejected_action)
                    rejected.append({
                        "robot_id": robot_id,
                        "task_id": rejected_cfg.task_id,
                        "reason": f"resolved_by:{resolver_name}",
                    })

            robot = self._robot_by_id[robot_id]
            adapted = robot.action_adapter.process(action)
            action_space = task_cfg.policy.action_space if task_cfg.policy is not None else self._infer_action_space(adapted)
            safe = robot.description.get_safety_filter().filter(adapted, action_space)
            final_actions[robot_id] = safe
        return final_actions, rejected

    def _step_multi(self, action: Optional[List[Action] | Dict[str, Action] | Action]):
        explicit_actions = self._normalize_explicit_actions(action)
        obs_by_robot = self.get_processed_obs_by_robot()
        candidate_actions = self._resolve_task_candidates(obs_by_robot, explicit_actions)
        final_actions, rejected = self._resolve_robot_actions(candidate_actions, obs_by_robot)

        ordered_actions = [
            final_actions.get(robot.robot_id, Action(joint_positions=robot.description.home_qpos))
            for robot in self._robots
        ]
        raw_obs_list = self._backend.step_multi(ordered_actions)
        next_obs_by_robot = {
            robot.robot_id: robot.obs_pipeline.process(raw_obs_list[idx])
            for idx, robot in enumerate(self._robots[: len(raw_obs_list)])
        }

        task_states, primary_obs, primary_reward, primary_done, primary_success, primary_failure = self.evaluate_active_tasks(
            next_obs_by_robot,
            final_actions,
        )

        primary_cfg = self._primary_task()
        info = EpisodeInfo(
            episode_id=self._episode_info.episode_id,
            step=task_states.get(primary_cfg.task_id, TaskStepState(primary_cfg.task_id)).step,
            reward=primary_reward,
            success=primary_success,
            failure=primary_failure,
        )
        info.extra["multi_agent"] = self._build_multi_info(next_obs_by_robot, final_actions, task_states, rejected)
        self._episode_info = info
        return primary_obs, primary_reward, primary_done, info

    def evaluate_active_tasks(
        self,
        obs_by_robot: dict[str, Observation],
        actions_by_robot: dict[str, Action],
    ) -> tuple[dict[str, TaskStepState], Observation, float, bool, bool, bool]:
        active_tasks = self._active_tasks()
        task_states: dict[str, TaskStepState] = {}
        primary_cfg = self._primary_task()
        primary_obs = obs_by_robot[primary_cfg.robot_ids[0]]
        primary_reward = 0.0
        primary_done = False
        primary_success = False
        primary_failure = False

        for task_cfg in active_tasks:
            task_cfg.task._on_step()
            primary_robot_id = task_cfg.robot_ids[0]
            task_obs = obs_by_robot[primary_robot_id]
            action_used = actions_by_robot.get(primary_robot_id, Action())
            reward = task_cfg.task.reward_fn(task_obs, action_used)
            success = task_cfg.task.success_fn(task_obs)
            failure = task_cfg.task.failure_fn(task_obs) or (task_cfg.task.step_count >= task_cfg.task.max_steps())
            done = success or failure
            task_states[task_cfg.task_id] = TaskStepState(
                task_id=task_cfg.task_id,
                robot_ids=list(task_cfg.robot_ids),
                reward=reward,
                done=done,
                success=success,
                failure=failure,
                step=task_cfg.task.step_count,
                active=True,
            )
            if task_cfg.task_id == primary_cfg.task_id:
                primary_obs = task_obs
                primary_reward = reward
                primary_done = done
                primary_success = success
                primary_failure = failure

        return task_states, primary_obs, primary_reward, primary_done, primary_success, primary_failure

    def _build_multi_info(
        self,
        obs_by_robot: dict[str, Observation],
        actions_by_robot: dict[str, Action],
        task_states: dict[str, TaskStepState],
        rejected: list[dict],
    ) -> MultiAgentInfo:
        multi_info = MultiAgentInfo(primary_task_id=self._primary_task().task_id)
        multi_info.arbitration_events = self._arbitrator.consume_events()

        for robot in self._robots:
            active_task_id = self._arbitrator.active_task_id(robot.robot_id)
            multi_info.active_tasks_by_robot[robot.robot_id] = active_task_id or ""
            multi_info.robot_states[robot.robot_id] = RobotStepState(
                robot_id=robot.robot_id,
                active_task_id=active_task_id or "",
                action=actions_by_robot.get(robot.robot_id),
                obs=obs_by_robot.get(robot.robot_id),
            )

        if task_states:
            multi_info.task_states.update(task_states)
        else:
            for task_cfg in self._tasks_cfg:
                multi_info.task_states[task_cfg.task_id] = TaskStepState(
                    task_id=task_cfg.task_id,
                    robot_ids=list(task_cfg.robot_ids),
                    active=task_cfg in self._active_tasks(),
                )

        multi_info.rejected_actions = rejected
        return multi_info

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

    @property
    def is_real(self) -> bool:
        return self._backend.is_real

    @property
    def description(self) -> RobotDescription:
        return self._description

    @property
    def backend(self) -> IBackend:
        return self._backend

    @property
    def task(self) -> ITask:
        return self._task

    @property
    def robots(self) -> List[RobotConfig]:
        return list(self._robots)

    @property
    def tasks(self) -> List[TaskConfig]:
        return list(self._tasks_cfg)

    @staticmethod
    def _infer_action_space(action: Action):
        from robodeploy.core.spaces import ActionSpace

        if action.joint_positions is not None:
            return ActionSpace.JOINT_POS
        if action.joint_velocities is not None:
            return ActionSpace.JOINT_VEL
        if action.joint_torques is not None:
            return ActionSpace.JOINT_TORQUE
        if action.ee_position is not None:
            return ActionSpace.CARTESIAN_POSE
        return ActionSpace.JOINT_POS

    def __repr__(self) -> str:
        if self._single_agent_mode:
            return f"RoboEnv(robot={self._description!r}, backend={self._backend!r}, real={self.is_real})"
        return f"RoboEnv(robots={len(self._robots)}, tasks={len(self._tasks_cfg)}, backend={self._backend!r}, real={self.is_real})"

