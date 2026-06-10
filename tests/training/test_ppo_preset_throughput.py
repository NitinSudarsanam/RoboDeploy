"""Honest coverage for preset PPO parallel-env throughput (GOAL 02).

``test_subproc_throughput_beats_sequential`` (in ``test_subproc_vec_env``) is the
only test that asserts >=3x speedup. It uses ``dummy_gym_env_factory`` on Linux
with ``start_method='fork'`` and is skipped on Windows because ``spawn`` IPC
overhead makes the ratio unreliable.

``robodeploy train ppo --preset kuka_pick_mujoco`` uses the same
``SubprocVecEnv`` worker model but defaults to ``start_method='spawn'`` (required
on Windows; also the CLI default on Linux). CI therefore does **not** claim
>=3x throughput for the preset/MuJoCo path—only that the CLI env factory is
picklable and steps correctly inside worker processes.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from functools import partial

import numpy as np
import pytest

from robodeploy.training.parallel_vec_env import SubprocVecEnv


def _has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


class PpoPresetThroughputTests(unittest.TestCase):
    def test_cli_env_factory_dummy_steps_in_subproc(self):
        """``train ppo --dummy`` path: factory must be spawn-picklable."""
        from robodeploy.cli import _make_gym_env_factory

        env_fn = _make_gym_env_factory(dummy=True, preset="", max_episode_steps=20)
        probe = env_fn()
        try:
            action_dim = int(probe.action_space.shape[0])
        finally:
            probe.close()
        vec = SubprocVecEnv([env_fn for _ in range(2)])
        try:
            obs, _ = vec.reset()
            self.assertEqual(len(obs), 2)
            action = np.zeros((2, action_dim), dtype=np.float32)
            next_obs, rewards, _, _, _ = vec.step(action)
            self.assertEqual(len(next_obs), 2)
            self.assertEqual(rewards.shape, (2,))
        finally:
            vec.close()

    @pytest.mark.skipif(not _has_mujoco(), reason="mujoco not installed")
    def test_cli_env_factory_preset_steps_in_subproc_spawn(self):
        """Preset factory must step under spawn (CLI default); no throughput claim."""
        from robodeploy.cli import _make_gym_env_factory

        env_fn = _make_gym_env_factory(dummy=False, preset="kuka_pick_mujoco", max_episode_steps=20)
        vec = SubprocVecEnv([env_fn for _ in range(2)], start_method="spawn")
        try:
            obs, _ = vec.reset()
            self.assertEqual(len(obs), 2)
            probe = env_fn()
            try:
                action_dim = int(probe.action_space.shape[0])
            finally:
                probe.close()
            action = np.zeros((2, action_dim), dtype=np.float32)
            _, rewards, _, _, _ = vec.step(action)
            self.assertEqual(rewards.shape, (2,))
        finally:
            vec.close()

    def test_cli_subproc_default_start_method_is_spawn(self):
        """Document that ``train ppo`` matches SubprocVecEnv spawn default."""
        from robodeploy.training.parallel_vec_env import SubprocVecEnv

        default = inspect.signature(SubprocVecEnv.__init__).parameters["start_method"].default
        self.assertEqual(default, "spawn")

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason=">=3x throughput ratio validated on Linux CI with fork, not spawn/Windows",
    )
    def test_fork_throughput_claim_scope_is_dummy_factory_only(self):
        """Fork speedup benchmark does not cover preset/MuJoCo or CLI spawn path."""
        import time

        from robodeploy.training.parallel_vec_env import dummy_gym_env_factory

        n_envs = 4
        n_steps = 20
        work_iters = 3000
        action = np.array([0.1], dtype=np.float32)

        seq_envs = [
            dummy_gym_env_factory(tag=i, max_steps=10_000, work_iters=work_iters) for i in range(n_envs)
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
            start_method="fork",
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
            msg=(
                f"dummy_gym_env_factory fork benchmark: expected >=3x, got {ratio:.2f}x "
                f"(preset CLI path uses spawn and is not covered by this claim)"
            ),
        )


if __name__ == "__main__":
    unittest.main()
