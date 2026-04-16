"""Run the user-defined Kuka sinusoid demo on MuJoCo."""

from __future__ import annotations

import time
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
        backend="mujoco",
        task="user_kuka_sinusoid",
        policy="user_sinusoid",
        backend_kwargs={"config": {"enable_viewer": True}},
        task_kwargs={"max_steps": 2000},
        policy_kwargs={"amplitude": 0.35, "frequency_hz": 0.2},
    )

    try:
        obs, info = env.reset()
    except ImportError as exc:
        print(exc)
        print("\nTo run this demo locally:\n  pip install mujoco\n  python -m examples.user_kuka_sinusoid.run_mujoco")
        return
    print("reset:", info)
    for i in range(1000):
        obs, reward, done, info = env.step()
        if i % 100 == 0:
            print("step", i, "q0", float(obs.joint_positions[0]))
        if done:
            break
        time.sleep(0.01)

    env.close()


if __name__ == "__main__":
    main()

