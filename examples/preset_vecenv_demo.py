"""Example: load preset names and build a SequentialVecEnv (contract demo)."""

from __future__ import annotations

from examples.config import load_example_preset, list_example_presets
from examples.vecenv_from_presets import vecenv_from_example_presets


def main() -> None:
    print("Example presets:", list_example_presets())
    preset = load_example_preset("kuka_pick_mujoco")
    print("Preset:", preset)
    print("Use examples.env_from_preset('kuka_pick_mujoco') for a single env.")
    vec = vecenv_from_example_presets(["kuka_pick_mujoco"])
    print("SequentialVecEnv env count:", vec.num_envs)


if __name__ == "__main__":
    main()
