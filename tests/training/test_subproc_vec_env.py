from __future__ import annotations

import sys
import time
import unittest
from functools import partial

import numpy as np
import pytest

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

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason=(
            "SubprocVecEnv throughput ratio is unreliable on Windows (spawn IPC overhead); "
            "validated on Linux CI with fork."
        ),
    )
    def test_subproc_throughput_beats_sequential(self):
        n_envs = 4
        n_steps = 40
        work_iters = 4000
        start_method = "fork" if sys.platform != "win32" else "spawn"
        action = np.array([0.1], dtype=np.float32)

        seq_envs = [
            dummy_gym_env_factory(tag=i, max_steps=10_000, work_iters=work_iters)
            for i in range(n_envs)
        ]
        t0 = time.perf_counter()
        for env in seq_envs:
            env.reset()
        for _ in range(n_steps):
            for env in seq_envs:
                env.step(action)
        seq_elapsed = time.perf_counter() - t0
        for env in seq_envs:
            env.close()

        vec = SubprocVecEnv(
            [
                partial(dummy_gym_env_factory, tag=i, max_steps=10_000, work_iters=work_iters)
                for i in range(n_envs)
            ],
            start_method=start_method,
        )
        try:
            t0 = time.perf_counter()
            vec.reset()
            batch = np.stack([action for _ in range(n_envs)], axis=0)
            for _ in range(n_steps):
                vec.step(batch)
            par_elapsed = time.perf_counter() - t0
        finally:
            vec.close()

        ratio = seq_elapsed / max(par_elapsed, 1e-9)
        self.assertGreater(
            ratio,
            3.0,
            msg=f"expected parallel >=3x sequential, got {ratio:.2f}x "
            f"(seq={seq_elapsed:.3f}s par={par_elapsed:.3f}s)",
        )


if __name__ == "__main__":
    unittest.main()
