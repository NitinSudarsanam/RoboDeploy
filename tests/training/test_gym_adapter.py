from __future__ import annotations

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from robodeploy.cli import _make_dummy_env
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action
from robodeploy.env import RoboEnv
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask
from robodeploy.training.gym_adapter import GymRoboEnv, observation_to_dict


class GymAdapterTests(unittest.TestCase):
    def test_five_tuple_step_and_truncation(self):
        env = GymRoboEnv(_make_dummy_env(), max_episode_steps=3)
        try:
            obs, info = env.reset()
            self.assertIn("proprio", obs)
            self.assertIsInstance(info, dict)
            terminated = False
            truncated = False
            for _ in range(4):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                self.assertIsInstance(reward, float)
            self.assertTrue(truncated or terminated)
        finally:
            env.close()

    def test_explicit_action_roundtrip(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        robo = RoboEnv(backend=DummyBackend(), robots=[robot])
        gym_env = GymRoboEnv(robo, max_episode_steps=5)
        try:
            obs, _ = gym_env.reset()
            target = np.array([0.5, -0.5], dtype=np.float32)
            obs, reward, terminated, truncated, info = gym_env.step(target)
            self.assertEqual(obs["proprio"].shape[0], 6)
            self.assertFalse(terminated)
        finally:
            gym_env.close()

    def test_observation_to_dict_proprio_dim(self):
        obs, _ = _make_dummy_env().reset()
        flat = observation_to_dict(obs, DummyTask().obs_spec())
        self.assertEqual(flat["proprio"].shape[0], 6)


if __name__ == "__main__":
    unittest.main()
