"""Franka Panda robomimic policy demo — sim or real.

Loads a robomimic checkpoint and runs it on the Franka Panda using the
RoboDeploy unified backend interface.  Switching between simulation and real
hardware is a **single line change** inside ``run()``.

State vector fed to the policy:
    [joint_positions(7), joint_velocities(7), gripper(1)] → 15-dim float32

Run (MuJoCo sim with viewer):
    conda activate ros2_env
    python -m robodeploy.demos.franka_robomimic_demo \\
        --checkpoint /path/to/policy.pth

Run (real hardware via ROS 2):
    conda activate ros2_env
    python -m robodeploy.demos.franka_robomimic_demo \\
        --checkpoint /path/to/policy.pth --real
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from robodeploy.policies.robomimic_policy import RobomimicPolicy

_DEFAULT_POLICY_HZ = 20.0
_CONTROL_HZ = 100.0
_TIMESTEP_S = 0.002
_DEFAULT_DURATION_S = 30.0


async def run(
    checkpoint_path: Path,
    use_real: bool,
    obs_key: str,
    policy_hz: float,
    action_smooth: float,
    duration_s: float,
    use_cpu: bool,
) -> None:
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

    policy = RobomimicPolicy(
        checkpoint_path=checkpoint_path,
        obs_key=obs_key,
        action_smooth=action_smooth,
        use_cuda=not use_cpu,
    )
    policy.start_episode()

    await engine.initialize()
    if not use_real:
        await engine.reset()

    print(f"[franka_robomimic_demo] backend={'real' if use_real else 'sim'}")
    print(f"[franka_robomimic_demo] checkpoint: {checkpoint_path}")
    print(f"[franka_robomimic_demo] policy_hz={policy_hz}  smooth={action_smooth}")
    print(f"[franka_robomimic_demo] running for {duration_s:.1f}s…")

    control_period_s = 1.0 / _CONTROL_HZ
    policy_period_s = 1.0 / max(policy_hz, 1e-3)

    start_time = time.perf_counter()
    next_policy_t = start_time
    step = 0

    while time.perf_counter() - start_time < duration_s:
        loop_start = time.perf_counter()

        obs = await engine.get_obs()

        # Only run the policy at policy_hz; hold the last action between calls
        if loop_start >= next_policy_t:
            action = policy.get_action(obs)
            next_policy_t = loop_start + policy_period_s

        await engine.apply_action(action)

        if step % 100 == 0:
            q0 = float(obs.joint_positions[0])
            ee_z = float(obs.ee_position[2])
            print(f"  step={step:04d}  q0={q0:+.3f}  ee_z={ee_z:.3f}")

        step += 1

        elapsed = time.perf_counter() - loop_start
        sleep_s = control_period_s - elapsed
        if sleep_s > 0.0:
            await asyncio.sleep(sleep_s)

    await engine.shutdown()
    print("[franka_robomimic_demo] done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a robomimic policy on the Franka Panda (sim or real)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to robomimic .pth checkpoint file.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use FrankaRealBackend (ROS 2 Jazzy) instead of MujocoEngine.",
    )
    parser.add_argument(
        "--obs-key",
        type=str,
        default="state",
        help="Observation dict key expected by the policy (default: %(default)s).",
    )
    parser.add_argument(
        "--policy-rate",
        type=float,
        default=_DEFAULT_POLICY_HZ,
        help="Policy inference rate in Hz (default %(default)s).",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.2,
        help="Action smoothing factor [0, 1] (default %(default)s).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=_DEFAULT_DURATION_S,
        help="Demo duration in seconds (default %(default)s).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (no CUDA).",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            checkpoint_path=args.checkpoint,
            use_real=args.real,
            obs_key=args.obs_key,
            policy_hz=args.policy_rate,
            action_smooth=args.smooth,
            duration_s=args.duration,
            use_cpu=args.cpu,
        )
    )


if __name__ == "__main__":
    main()
