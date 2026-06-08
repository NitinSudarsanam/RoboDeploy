"""RoboBridge — process-owned control bridge for real hardware."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import time
from typing import Callable, Optional

from robodeploy.action_trajectory import ActionTrajectory, ActionTrajectorySpec
from robodeploy.core.types import Action, EpisodeInfo
from robodeploy.env import RoboEnv


EnvFactory = Callable[[], RoboEnv]


class LatencyModel:
    """Models comm + execution delay for control-process action interpolation."""

    def __init__(
        self,
        *,
        mean_delay_s: float = 0.02,
        jitter_std_s: float = 0.005,
        max_buffer: int = 32,
        seed: int | None = None,
    ) -> None:
        import numpy as np

        self.mean_delay_s = float(mean_delay_s)
        self.jitter_std_s = float(jitter_std_s)
        self.max_buffer = max(1, int(max_buffer))
        self._rng = np.random.default_rng(seed)

    def predict_execution_time(self, command_time: float) -> float:
        jitter = float(self._rng.normal(0.0, self.jitter_std_s))
        return float(command_time) + self.mean_delay_s + jitter

    def interpolate_command(self, buffer: list[tuple[float, Action]], now: float) -> Action | None:
        """Linearly interpolate between buffered (timestamp, action) pairs."""
        if not buffer:
            return None
        target_t = now - self.mean_delay_s
        if target_t <= buffer[0][0]:
            return buffer[0][1]
        if target_t >= buffer[-1][0]:
            return buffer[-1][1]
        for i in range(1, len(buffer)):
            t0, a0 = buffer[i - 1]
            t1, a1 = buffer[i]
            if t0 <= target_t <= t1:
                if a0.joint_positions is None or a1.joint_positions is None:
                    return a1
                import numpy as np

                alpha = (target_t - t0) / max(1e-9, t1 - t0)
                q0 = np.asarray(a0.joint_positions, dtype=np.float32)
                q1 = np.asarray(a1.joint_positions, dtype=np.float32)
                q = q0 + alpha * (q1 - q0)
                return Action(joint_positions=q, gripper=a1.gripper)
        return buffer[-1][1]


class EStopFlag:
    """Shared e-stop/pause flag for bridge processes."""

    def __init__(self, event=None) -> None:  # noqa: ANN001
        self._event = event or mp.get_context("spawn").Event()

    @property
    def raw(self):  # noqa: ANN201
        return self._event

    def trigger(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    @property
    def active(self) -> bool:
        return bool(self._event.is_set())


class RoboBridge:
    """Real-hardware bridge with multi-robot parity.

    Inference happens at the env's natural rate via env.step(); the
    control loop replays the latest action at the higher control_hz to keep
    the hardware drivers happy.
    """

    def __init__(
        self,
        env: RoboEnv,
        *,
        env_factory: EnvFactory | None = None,
        control_hz: Optional[float] = None,
        verbose: bool = False,
        latency_model: LatencyModel | None = None,
    ) -> None:
        if not env.is_real:
            raise ValueError(
                "RoboBridge is for real hardware (env.is_real == True). "
                "For sim, call env.step() in a sync loop."
            )

        self._env = env
        self._env_factory = env_factory
        self._verbose = verbose
        self._control_hz = float(control_hz or env.backend.control_hz)
        self._latency_model = latency_model or LatencyModel()
        self._action_buffer: list[tuple[float, Action]] = []
        self._trajectory: Optional[ActionTrajectory] = None
        self._control_proc: Optional[mp.Process] = None
        self._obs_queue = None
        self._estop: EStopFlag | None = None

        env.set_pause_hooks(
            on_pause=self.pause,
            on_resume=self.resume,
        )

    async def __aenter__(self) -> "RoboBridge":
        if self._env_factory is None:
            raise RuntimeError(
                "RoboBridge now requires env_factory so the control process can own "
                "the backend. Pass a top-level picklable callable returning a fresh RoboEnv."
            )
        self._start_control_process()
        return self

    async def __aexit__(self, *_) -> None:
        self._stop_control_process()
        try:
            self._env.close()
        except Exception:
            pass

    async def run(self, max_steps: Optional[int] = None) -> EpisodeInfo:
        primary_task = self._env.primary_robot.tasks[self._env.primary_robot.active_task_id]
        limit = max_steps or primary_task.task.max_steps()
        step = 0
        info = EpisodeInfo()

        while step < limit:
            t_start = time.perf_counter()
            obs_by_robot = self._read_obs_snapshot(timeout_s=2.0)
            actions = {
                robot.robot_id: robot.step(obs_by_robot[robot.robot_id])
                for robot in self._env.robots
                if robot.robot_id in obs_by_robot
            }
            now = time.perf_counter()
            for rid, act in actions.items():
                assert self._trajectory is not None
                self._action_buffer.append((now, act))
                if len(self._action_buffer) > self._latency_model.max_buffer:
                    self._action_buffer.pop(0)
                delayed = self._latency_model.interpolate_command(self._action_buffer, now)
                self._trajectory.write(rid, delayed if delayed is not None else act)

            step += 1
            if step >= limit:
                break

            target_hz = self._inference_hz()
            elapsed = time.perf_counter() - t_start
            remaining = (1.0 / target_hz) - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            elif self._verbose:
                print(f"[InferenceLoop] {(-remaining) * 1000:.1f}ms over budget")

        return info

    def pause(self) -> None:
        if self._estop is not None:
            self._estop.trigger()

    def resume(self) -> None:
        if self._estop is not None:
            self._estop.clear()

    @property
    def control_hz(self) -> float:
        return self._control_hz

    @property
    def estop_active(self) -> bool:
        return bool(self._estop and self._estop.active)

    def _inference_hz(self) -> float:
        requested = 0.0
        for robot in self._env.robots:
            for task_id in robot.active_task_ids():
                robot_task = robot.tasks.get(task_id)
                if robot_task is None:
                    continue
                for policy in robot_task.policies.values():
                    requested = max(requested, float(getattr(policy, "action_hz", 0.0) or 0.0))
        if requested <= 0.0:
            return self.control_hz
        return min(self.control_hz, requested)

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
        self._obs_queue = ctx.Queue(maxsize=1)
        self._estop = EStopFlag(ctx.Event())
        self._control_proc = ctx.Process(
            target=_control_process_entry,
            args=(
                self._env_factory,
                self._trajectory.name,
                spec,
                float(self.control_hz),
                self._obs_queue,
                self._estop.raw,
                self._verbose,
            ),
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
        self._obs_queue = None

    def _read_obs_snapshot(self, *, timeout_s: float) -> dict:
        if self._obs_queue is None:
            raise RuntimeError("RoboBridge control process has not been started.")
        try:
            msg = self._obs_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            if self._estop is not None:
                self._estop.trigger()
            raise TimeoutError("Timed out waiting for control-process observation snapshot.") from exc
        if isinstance(msg, dict) and "error" in msg:
            raise RuntimeError(str(msg["error"]))
        return dict(msg)


def _control_process_entry(
    env_factory: EnvFactory,
    shm_name: str,
    spec: ActionTrajectorySpec,
    hz: float,
    obs_queue,
    estop_event,
    verbose: bool,
) -> None:
    traj = ActionTrajectory(spec, name=shm_name, create=False)
    period = 1.0 / max(1.0, float(hz))
    env = None
    try:
        env = env_factory()
        env.reset()
        _publish_obs_snapshot(env, obs_queue)
        while True:
            t0 = time.perf_counter()
            actions: list[Action] = []
            for robot in env.robots:
                if estop_event.is_set():
                    actions.append(Action(joint_positions=robot.description.home_qpos))
                    continue
                try:
                    q, _ = traj.read_latest_joint_positions(robot.robot_id)
                except TimeoutError:
                    estop_event.set()
                    q = robot.description.home_qpos
                actions.append(Action(joint_positions=q if q is not None else robot.description.home_qpos))
            env.backend.step_multi(actions)
            _publish_obs_snapshot(env, obs_queue)
            dt = time.perf_counter() - t0
            rem = period - dt
            if rem > 0:
                time.sleep(rem)
            elif verbose:
                print(f"[ControlLoopProcess] {(-rem) * 1000:.1f}ms overrun")
    except Exception as exc:
        try:
            obs_queue.put_nowait({"error": repr(exc)})
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        traj.close()


def _publish_obs_snapshot(env: RoboEnv, obs_queue) -> None:  # noqa: ANN001
    obs = env.get_processed_obs_by_robot()
    try:
        while True:
            obs_queue.get_nowait()
    except Exception:
        pass
    obs_queue.put(obs)
