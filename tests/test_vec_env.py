from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.types import Action, EpisodeInfo, Observation
from robodeploy.vec_env import SequentialVecEnv


class _MockEnv:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.steps = 0

    def reset(self):
        obs = Observation(
            joint_positions=np.array([float(self.tag == "a")], dtype=np.float32),
            joint_velocities=np.zeros(1, dtype=np.float32),
            joint_torques=np.zeros(1, dtype=np.float32),
            ee_position=np.zeros(3, dtype=np.float32),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ee_velocity=np.zeros(3, dtype=np.float32),
            ee_angular_velocity=np.zeros(3, dtype=np.float32),
        )
        return obs, EpisodeInfo()

    def step(self, action):  # noqa: ANN001
        self.steps += 1
        obs, _ = self.reset()
        return obs, float(self.steps), self.steps >= 2, EpisodeInfo()


class VecEnvTests(unittest.TestCase):
    def test_sequential_vec_env_steps_all_envs(self):
        envs = [_MockEnv("a"), _MockEnv("b")]
        vec = SequentialVecEnv(envs)
        obs_list, _ = vec.reset()
        self.assertEqual(len(obs_list), 2)
        next_obs, rewards, dones, _ = vec.step([Action(), Action()])
        self.assertEqual(len(next_obs), 2)
        self.assertEqual(rewards, [1.0, 1.0])
        self.assertEqual(dones, [False, False])


if __name__ == "__main__":
    unittest.main()
