from __future__ import annotations

import unittest

from robodeploy.cli import _make_dummy_env
from robodeploy.cli_helpers import action_fn_for_mode, close_quietly
from robodeploy.training.rollout import RolloutCollector


class RolloutCollectorTests(unittest.TestCase):
    def test_collect_episode_records_frames(self):
        env = _make_dummy_env()
        try:
            collector = RolloutCollector(env, max_steps=5)
            frames = collector.collect_episode(action_fn_for_mode("hold", env))
            self.assertGreater(len(frames), 0)
            self.assertIn("joint_positions", frames[0].action)
        finally:
            close_quietly(env)

    def test_collect_n_episodes(self):
        env = _make_dummy_env()
        try:
            recorder = RolloutCollector(env, max_steps=3).collect_n(
                2,
                action_fn_for_mode("zero", env),
            )
            self.assertGreaterEqual(len(recorder.frames), 2)
        finally:
            close_quietly(env)


if __name__ == "__main__":
    unittest.main()
