from __future__ import annotations

import unittest
from functools import partial

import numpy as np

from robodeploy.training.parallel_vec_env import SubprocVecEnv, dummy_gym_env_factory


class SubprocVecEnvTests(unittest.TestCase):
    def test_subproc_vec_env_steps_all_workers(self):
        n = 4
        vec = SubprocVecEnv([partial(dummy_gym_env_factory, tag=i) for i in range(n)])
        try:
            obs, infos = vec.reset()
            self.assertEqual(len(obs), n)
            self.assertEqual(len(infos), n)
            actions = np.stack([np.array([0.1], dtype=np.float32) for _ in range(n)], axis=0)
            next_obs, rewards, terminated, truncated, step_infos = vec.step(actions)
            self.assertEqual(len(next_obs), n)
            self.assertEqual(rewards.shape, (n,))
            self.assertEqual(terminated.shape, (n,))
            self.assertEqual(len(step_infos), n)
        finally:
            vec.close()

    def test_subproc_reset_and_multi_step(self):
        n = 2
        vec = SubprocVecEnv([partial(dummy_gym_env_factory, tag=i, max_steps=10) for i in range(n)])
        try:
            vec.reset(seeds=[0, 1])
            for _ in range(5):
                _, rewards, terminated, truncated, _ = vec.step(
                    [np.array([0.2], dtype=np.float32) for _ in range(n)]
                )
                self.assertEqual(rewards.shape, (n,))
                if terminated.all() or truncated.all():
                    break
        finally:
            vec.close()


if __name__ == "__main__":
    unittest.main()
