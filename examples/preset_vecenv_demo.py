"""Example: load a preset name and step a SequentialVecEnv (contract demo).

Uses DummyBackend stand-in when run without assets; intended as API documentation.
"""

from __future__ import annotations

from robodeploy.config import load_preset
from robodeploy.vec_env import SequentialVecEnv


def main() -> None:
    preset = load_preset("kuka_pick_mujoco")
    print("Preset:", preset)
    print("Use RoboEnv.from_preset('kuka_pick_mujoco') for a single env.")
    print("Wrap multiple RoboEnv instances in SequentialVecEnv for batched stepping.")


if __name__ == "__main__":
    main()
