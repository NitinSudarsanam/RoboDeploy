from __future__ import annotations

import unittest

import numpy as np

from robodeploy.training.parallel_vec_env import AsyncVecEnv, dummy_gym_env_factory


class AsyncVecEnvTests(unittest.TestCase):
    def test_async_vec_env_step_sync(self):
        vec = AsyncVecEnv([lambda: dummy_gym_env_factory(tag=i) for i in range(2)])
        try:
            obs, infos = vec.reset_sync()
            self.assertEqual(len(obs), 2)
            actions = [np.array([0.1], dtype=np.float32) for _ in range(2)]
            next_obs, rewards, terminated, truncated, step_infos = vec.step_sync(actions)
            self.assertEqual(len(next_obs), 2)
            self.assertEqual(rewards.shape, (2,))
            self.assertEqual(len(step_infos), 2)
        finally:
            vec.close()


if __name__ == "__main__":
    unittest.main()
