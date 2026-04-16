"""Run the user-defined Kuka sinusoid demo on Isaac Sim (if installed)."""

from __future__ import annotations

import sys
from pathlib import Path

def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from robodeploy import RoboEnv  # noqa: E402

# Import registers @register_* components.
from examples.user_kuka_sinusoid import components  # noqa: E402,F401


def main() -> None:
    env = RoboEnv.make(
        robot="user_kuka",
        backend="isaacsim",
        task="user_kuka_sinusoid",
        policy="user_sinusoid",
        backend_kwargs={"config": {
            # Use a lighter experience by default to avoid optional extensions
            # failing to load on some Windows GPU/driver setups.
            "experience": "isaacsim.exp.base.python.kit",
            "headless": False,
            "renderer": "RaytracedLighting",
        }},
        task_kwargs={"max_steps": 2000},
        policy_kwargs={"amplitude": 0.35, "frequency_hz": 0.2},
    )

    env.reset()
    for _ in range(1000):
        env.step()
    env.close()


if __name__ == "__main__":
    main()

