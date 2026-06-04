"""Build a SequentialVecEnv from example YAML presets."""

from __future__ import annotations

from robodeploy.vec_env import SequentialVecEnv

from examples.env_from_preset import env_from_preset


def vecenv_from_example_presets(preset_names: list[str], **overrides) -> SequentialVecEnv:
    envs = [env_from_preset(name, robot_id=f"robot{i}", **overrides) for i, name in enumerate(preset_names)]
    return SequentialVecEnv(envs)


def main() -> None:
    vec = vecenv_from_example_presets(["kuka_pick_mujoco"])
    print("Built SequentialVecEnv with", vec.num_envs, "env(s).")


if __name__ == "__main__":
    main()
