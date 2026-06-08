from __future__ import annotations

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robodeploy.training.replay_buffer import ReplayBuffer, RolloutBuffer


class ReplayBufferTests(unittest.TestCase):
    def test_replay_buffer_uniform_sample(self):
        buf = ReplayBuffer(100, obs_dim=4, action_dim=2)
        for i in range(20):
            obs = np.full(4, float(i), dtype=np.float32)
            act = np.array([i, i], dtype=np.float32)
            buf.add(obs, act, float(i), obs + 0.1, i % 5 == 0)
        self.assertEqual(buf.size, 20)
        batch = buf.sample(8)
        self.assertEqual(len(batch), 5)
        self.assertEqual(batch[0].shape, (8, 4))

    def test_rollout_buffer_minibatches(self):
        buf = RolloutBuffer(capacity=16)
        for i in range(8):
            buf.add(
                obs={"proprio": np.array([float(i)], dtype=np.float32)},
                action=np.array([i], dtype=np.float32),
                reward=float(i),
                value=float(i),
                log_prob=-0.1,
                done=i == 7,
            )
        buf.advantages = [0.0] * buf.size
        buf.returns = [float(i) for i in range(buf.size)]
        batches = list(buf.get(minibatch_size=3))
        self.assertGreaterEqual(len(batches), 2)
        self.assertEqual(batches[0]["actions"].shape[1], 1)


if __name__ == "__main__":
    unittest.main()
