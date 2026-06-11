"""One-command visual pick-place rehearsal across simulators.

Edit ``SIMULATOR`` or pass ``--simulator`` / ``--headless`` on the CLI.

    python -m examples.kuka_ft_imu_pick.run_visual
    python -m examples.kuka_ft_imu_pick.run_visual --simulator mujoco --seed 0
    python -m examples.kuka_ft_imu_pick.run_visual --simulator ros2_rviz   # Linux/WSL + ROS Jazzy
    python -m examples.kuka_ft_imu_pick.run_visual --simulator gazebo      # Linux/WSL + gz sim

Platform notes:
  - MuJoCo viewer: Windows, Linux, macOS (``--viewer`` default).
  - RViz / Gazebo GUI: Ubuntu 22.04+ with ROS 2 Jazzy, DISPLAY or WSLg.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

SIMULATOR = "mujoco"
SEED = 0
STEPS = 1500


def _warn_display(simulator: str, headless: bool) -> None:
    if headless or simulator == "mujoco":
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    print(
        f"WARNING: {simulator} visual mode but DISPLAY/WAYLAND_DISPLAY unset. "
        "Use WSLg/X11 on WSL or pass --headless.",
        file=sys.stderr,
    )


def _linux_ubuntu_release() -> str | None:
    if sys.platform != "linux":
        return None
    try:
        out = subprocess.check_output(
            ["lsb_release", "-rs"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        return out.strip() or None
    except Exception:
        return None


def _warn_platform(simulator: str) -> None:
    if simulator == "mujoco":
        return
    if sys.platform == "win32":
        print(
            f"WARNING: {simulator} visual demo is not supported on native Windows. "
            "Use WSL2 Ubuntu 24.04 + ROS Jazzy (`wsl -d Ubuntu`).",
            file=sys.stderr,
        )
        return
    ubuntu = _linux_ubuntu_release()
    if simulator in {"ros2_rviz", "gazebo"}:
        if ubuntu is None:
            print(
                "NOTE: Ubuntu 24.04 + Jazzy recommended for RViz/Gazebo visual picks.",
                file=sys.stderr,
            )
        elif not ubuntu.startswith("24."):
            print(
                f"NOTE: Ubuntu 24.04 + Jazzy recommended; detected Ubuntu {ubuntu}.",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visual Kuka FT+IMU pick-place rehearsal.")
    parser.add_argument(
        "--simulator",
        default=SIMULATOR,
        choices=("mujoco", "gazebo", "ros2_rviz", "rviz"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--headless", action="store_true", help="Force headless backends.")
    args = parser.parse_args(argv)

    sim = "ros2_rviz" if args.simulator == "rviz" else args.simulator
    _warn_platform(sim)
    _warn_display(sim, args.headless)

    cmd = [
        sys.executable,
        "-m",
        "examples.cli",
        "run-episode",
        "--preset",
        "kuka_ft_imu_pick",
        "--simulator",
        sim,
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--json",
    ]
    if args.headless:
        cmd.append("--headless")
    elif sim == "mujoco":
        cmd.append("--viewer")

    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
