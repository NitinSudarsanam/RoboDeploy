"""RoboBridge — decoupled control/inference bridge for real hardware."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from robodeploy.core.types import Action, EpisodeInfo
from robodeploy.env import RoboEnv


class ActionBuffer:
    """Thread-safe store for latest per-robot actions."""

    def __init__(self, freeze_actions: Optional[dict[str, Action]] = None) -> None:
        self._actions: dict[str, Action] = dict(freeze_actions or {})
        self._lock = threading.Lock()
        self._write_count = 0

    def put(self, actions: dict[str, Action]) -> None:
        with self._lock:
            self._actions.update(actions)
            self._write_count += 1

    def get(self) -> dict[str, Action]:
        with self._lock:
            return dict(self._actions)

    @property
    def has_action(self) -> bool:
        with self._lock:
            return bool(self._actions)

    @property
    def write_count(self) -> int:
        with self._lock:
            return self._write_count


class ControlLoop:
    """High-frequency loop that sends the latest per-robot actions."""

    def __init__(
        self,
        env: RoboEnv,
        buffer: ActionBuffer,
        control_hz: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        self._env = env
        self._buffer = buffer
        self._hz = control_hz or env.backend.control_hz
        self._period = 1.0 / self._hz
        self._verbose = verbose
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ControlLoop")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def pause(self) -> None:
        self._paused = True
        if self._verbose:
            print("[ControlLoop] Paused.")

    def resume(self) -> None:
        self._paused = False
        if self._verbose:
            print("[ControlLoop] Resumed.")

    def _loop(self) -> None:
        while self._running:
            t_start = time.perf_counter()

            if not self._paused:
                actions_by_robot = self._buffer.get()
                if actions_by_robot:
                    ordered_actions = [
                        actions_by_robot.get(robot.robot_id, Action(joint_positions=robot.description.home_qpos))
                        for robot in self._env.robots
                    ]
                    if len(ordered_actions) == 1:
                        self._env.backend.step(ordered_actions[0])
                    else:
                        self._env.backend.step_multi(ordered_actions)

            elapsed = time.perf_counter() - t_start
            remaining = self._period - elapsed
            if remaining > 0:
                time.sleep(remaining)
            elif self._verbose and not self._paused:
                print(f"[ControlLoop] {(-remaining) * 1000:.1f}ms overrun")


class RoboBridge:
    """Real-hardware bridge with multi-agent parity for supported modes."""

    def __init__(
        self,
        env: RoboEnv,
        control_hz: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        if not env.is_real:
            raise ValueError(
                "RoboBridge is for real hardware (env.is_real == True). "
                "For sim, use RoboEnv.step() in a sync loop."
            )
        if not any(task_cfg.policy is not None for task_cfg in env.tasks):
            raise ValueError("RoboBridge requires at least one TaskConfig with a policy.")

        self._env = env
        self._verbose = verbose
        self._buffer = ActionBuffer()
        self._control = ControlLoop(env, self._buffer, control_hz, verbose)

        env.set_pause_hooks(
            on_pause=self._control.pause,
            on_resume=self._control.resume,
        )

    async def __aenter__(self) -> "RoboBridge":
        self._env.reset()
        self._control.start()
        return self

    async def __aexit__(self, *_) -> None:
        self._control.stop()
        self._env.close()

    async def run(self, max_steps: Optional[int] = None) -> EpisodeInfo:
        primary_task = self._env.tasks[0]
        limit = max_steps or primary_task.task.max_steps()
        step = 0
        _, info = self._env.reset()

        while step < limit:
            t_start = time.perf_counter()
            obs_by_robot = self._env.get_processed_obs_by_robot()
            candidate_actions = self._env._resolve_task_candidates(obs_by_robot)
            final_actions, rejected = self._env._resolve_robot_actions(candidate_actions, obs_by_robot)
            self._buffer.put(final_actions)
            task_states, _, primary_reward, primary_done, primary_success, primary_failure = self._env.evaluate_active_tasks(
                obs_by_robot,
                final_actions,
            )
            multi_info = self._env._build_multi_info(obs_by_robot, final_actions, task_states, rejected)
            info = EpisodeInfo(
                episode_id=info.episode_id,
                step=step + 1,
                reward=primary_reward,
                success=primary_success,
                failure=primary_failure,
                extra={"multi_agent": multi_info},
            )

            step += 1
            if primary_done or step >= limit:
                break

            elapsed = time.perf_counter() - t_start
            remaining = (1.0 / self._control._hz) - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            elif self._verbose:
                print(f"[InferenceLoop] {(-remaining) * 1000:.1f}ms over budget")

        return info

    @property
    def control_hz(self) -> float:
        return self._control._hz

    @property
    def buffer(self) -> ActionBuffer:
        return self._buffer

