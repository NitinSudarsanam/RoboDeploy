"""Live Gazebo sensor rig demo (mirrors sensor-live-gazebo CI smoke)."""

from __future__ import annotations

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()
import time
from pathlib import Path



from examples.config import PRESETS_FILE, load_example_preset  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EMPTY_WORLD = REPO_ROOT / "tests" / "fixtures" / "gazebo_empty.sdf"


def main() -> None:
    cfg = load_example_preset("kuka_sensor_gazebo")
    cfg = {
        **cfg,
        "max_episode_steps": 200,
        "backend_kwargs": {
            "config": {
                "sim": {
                    "kind": "gazebo",
                    "world": str(EMPTY_WORLD),
                    "headless": True,
                    "readiness_timeout_s": 45.0,
                    "wait_for_topics": [],
                }
            }
        },
    }
    try:
        env = RoboEnv.from_config(cfg)
    except Exception as exc:
        print("Failed to start Gazebo backend:", exc)
        print(f"\nRequires gz on PATH and ROS2. Presets: {PRESETS_FILE}")
        return

    try:
        obs, info = env.reset()
        print("reset episode", info.episode_id)
        for i in range(200):
            obs, reward, done, info = env.step()
            if i % 20 == 0:
                print(
                    f"step {i:3d} reward={reward:6.3f} "
                    f"img={'yes' if obs.images.get('wrist_camera') is not None else 'no'} "
                    f"ft={obs.ft_forces.get('wrist_ft')}"
                )
            if done:
                break
            time.sleep(0.02)
    finally:
        env.close()


if __name__ == "__main__":
    main()
