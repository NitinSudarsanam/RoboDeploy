"""RoboBridge — decoupled control/inference bridge for real hardware.

Minimal multi-robot bridge that wraps a `RoboEnv`. The high-rate control loop
runs in a background thread (best-effort; a process-based variant is sketched
below). Each tick the bridge calls `env.step()` and publishes resulting actions
to an `ActionBuffer` for the control loop to consume.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import threading
import time
from typing import Optional

from robodeploy.action_trajectory import ActionTrajectory, ActionTrajectorySpec
from robodeploy.core.types import Action, EpisodeInfo
from robodeploy.env import RoboEnv


class ActionBuffer:
    """Thread-safe store for the latest per-robot action."""

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
    """High-frequency loop that re-publishes the latest action to the backend."""

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
                    self._env.backend.step_multi(ordered_actions)

            elapsed = time.perf_counter() - t_start
            remaining = self._period - elapsed
            if remaining > 0:
                time.sleep(remaining)
            elif self._verbose and not self._paused:
                print(f"[ControlLoop] {(-remaining) * 1000:.1f}ms overrun")


class RoboBridge:
    """Real-hardware bridge with multi-robot parity.

    Inference happens at the env's natural rate via env.step(); the
    control loop replays the latest action at the higher control_hz to keep
    the hardware drivers happy.
    """

    def __init__(
        self,
        env: RoboEnv,
        control_hz: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        if not env.is_real:
            raise ValueError(
                "RoboBridge is for real hardware (env.is_real == True). "
                "For sim, call env.step() in a sync loop."
            )

        self._env = env
        self._verbose = verbose
        self._buffer = ActionBuffer()
        self._control = ControlLoop(env, self._buffer, control_hz, verbose)
        self._trajectory: Optional[ActionTrajectory] = None
        self._control_proc: Optional[mp.Process] = None

        env.set_pause_hooks(
            on_pause=self._control.pause,
            on_resume=self._control.resume,
        )

    async def __aenter__(self) -> "RoboBridge":
        self._env.reset()
        try:
            self._start_control_process()
        except Exception as exc:
            if self._verbose:
                print(f"[RoboBridge] Process control loop unavailable, falling back to thread loop: {exc}")
            self._control.start()
        return self

    async def __aexit__(self, *_) -> None:
        self._stop_control_process()
        self._control.stop()
        self._env.close()

    async def run(self, max_steps: Optional[int] = None) -> EpisodeInfo:
        primary_task = self._env.primary_robot.tasks[self._env.primary_robot.active_task_id]
        limit = max_steps or primary_task.task.max_steps()
        step = 0
        _, info = self._env.reset()

        while step < limit:
            t_start = time.perf_counter()
            obs, reward, done, info = self._env.step()
            multi_info = info.extra.get("multi_agent")
            if multi_info is not None:
                actions = {
                    rid: state.action
                    for rid, state in multi_info.robot_states.items()
                    if state.action is not None
                }
                if actions:
                    if self._trajectory is not None:
                        for rid, act in actions.items():
                            self._trajectory.write(rid, act)
                    else:
                        self._buffer.put(actions)

            step += 1
            if done or step >= limit:
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

    # ------------------------------------------------------------------
    # Process-based control loop (best-effort)
    # ------------------------------------------------------------------

    def _start_control_process(self) -> None:
        spec = ActionTrajectorySpec(
            robot_ids=[r.robot_id for r in self._env.robots],
            dof_by_robot={r.robot_id: int(r.description.dof) for r in self._env.robots},
        )
        self._trajectory = ActionTrajectory(spec)
        ctx = mp.get_context("spawn")
        self._control_proc = ctx.Process(
            target=_control_process_entry,
            args=(self._trajectory.name, spec, float(self._control._hz), self._verbose),
            daemon=True,
            name="RoboBridgeControlLoop",
        )
        self._control_proc.start()

    def _stop_control_process(self) -> None:
        if self._control_proc is not None:
            try:
                self._control_proc.terminate()
            except Exception:
                pass
            try:
                self._control_proc.join(timeout=2.0)
            except Exception:
                pass
            self._control_proc = None
        if self._trajectory is not None:
            try:
                self._trajectory.close()
                self._trajectory.unlink()
            except Exception:
                pass
            self._trajectory = None


def _control_process_entry(shm_name: str, spec: ActionTrajectorySpec, hz: float, verbose: bool) -> None:
    traj = ActionTrajectory(spec, name=shm_name, create=False)
    period = 1.0 / max(1.0, float(hz))
    try:
        while True:
            t0 = time.perf_counter()
            for rid in spec.robot_ids:
                _q, _ = traj.read_latest_joint_positions(rid)
                del _q
            dt = time.perf_counter() - t0
            rem = period - dt
            if rem > 0:
                time.sleep(rem)
            elif verbose:
                print(f"[ControlLoopProcess] {(-rem) * 1000:.1f}ms overrun")
    finally:
        traj.close()
