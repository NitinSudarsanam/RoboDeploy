"""Franka Panda oscillation demo — sim or real, viewer-enabled.

Demonstrates sinusoidal joint-space control on the Franka Panda using the
RoboDeploy unified backend interface.  Switching between simulation and real
hardware is a **single line change** at the top of ``main()``.

Run (MuJoCo sim with viewer):
    conda activate ros2_env
    python -m robodeploy.demos.franka_sim_viewer_demo

Run (real hardware via ROS 2):
    conda activate ros2_env
    python -m robodeploy.demos.franka_sim_viewer_demo --real
"""

from __future__ import annotations

import argparse
import asyncio
import time

from robodeploy.tasks.panda_oscillation import PandaOscillationTask

# Simulation default: 10 seconds of oscillation at 100 Hz = 1000 steps
_DEFAULT_DURATION_S = 10.0
_CONTROL_HZ = 100.0
_TIMESTEP_S = 0.002


async def run(use_real: bool, duration_s: float) -> None:
    # -----------------------------------------------------------------------
    # Engine selection — change this one line to switch between sim and real.
    # -----------------------------------------------------------------------
    if use_real:
        from robodeploy.backends.real import FrankaRealBackend
        engine = FrankaRealBackend(robots=["franka"])
    else:
        from robodeploy.backends.sim import MujocoEngine
        engine = MujocoEngine(
            robots=["franka"],
            config={"timestep": _TIMESTEP_S, "control_hz": _CONTROL_HZ},
            enable_viewer=True,
        )
    # -----------------------------------------------------------------------

    total_steps = int(duration_s * _CONTROL_HZ)
    task = PandaOscillationTask(
        robot_id=0,
        timestep_s=_TIMESTEP_S,
        max_steps=total_steps,
    )

    await engine.initialize()
    if not use_real:
        await engine.reset()
    task.reset()

    print(f"[franka_sim_viewer_demo] backend={'real' if use_real else 'sim'}")
    print(f"[franka_sim_viewer_demo] instruction: {task.get_instruction()}")
    print(f"[franka_sim_viewer_demo] running {total_steps} steps ({duration_s:.1f}s)…")

    period_s = 1.0 / _CONTROL_HZ
    step = 0
    while not task.is_done():
        loop_start = time.perf_counter()

        obs = await engine.get_obs(tasks=[task])
        action = task.next_action(obs)
        await engine.apply_action(action)

        if step % 100 == 0:
            q0 = float(obs.joint_positions[0])
            grip = obs.gripper_state
            print(
                f"  step={step:04d}  q0={q0:+.3f} rad  "
                f"gripper={grip:.2f}" if grip is not None else f"  step={step:04d}  q0={q0:+.3f} rad"
            )

        step += 1

        # Real-time pacing (sim: MJX runs faster than wall-clock; real: safety guard)
        elapsed = time.perf_counter() - loop_start
        sleep_s = period_s - elapsed
        if sleep_s > 0.0:
            await asyncio.sleep(sleep_s)

    await engine.shutdown()
    print("[franka_sim_viewer_demo] done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Franka Panda sinusoidal oscillation demo (sim or real)."
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use FrankaRealBackend (ROS 2 Jazzy) instead of MujocoEngine.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=_DEFAULT_DURATION_S,
        help="Demo duration in seconds (default %(default)s).",
    )
    args = parser.parse_args()

    asyncio.run(run(use_real=args.real, duration_s=args.duration))


if __name__ == "__main__":
    main()
