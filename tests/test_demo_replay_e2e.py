from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.types import Action, EpisodeInfo, Observation
from robodeploy.demo_recording import DemoSession


def make_obs(step: int) -> Observation:
    return Observation(
        joint_positions=np.array([float(step), 0.0], dtype=np.float32),
        joint_velocities=np.zeros(2, dtype=np.float32),
        joint_torques=np.zeros(2, dtype=np.float32),
        ee_position=np.zeros(3, dtype=np.float32),
        ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ee_velocity=np.zeros(3, dtype=np.float32),
        ee_angular_velocity=np.zeros(3, dtype=np.float32),
    )


class _MockRoboEnv:
  """Minimal env stand-in for record/replay contract tests."""

  def __init__(self) -> None:
    self._step = 0
    self.replayed: list[list[float]] = []

  def reset(self):
    self._step = 0
    return make_obs(0), EpisodeInfo()

  def step(self, action: Action | None = None):
    if action is not None and action.joint_positions is not None:
      self.replayed.append([float(v) for v in action.joint_positions])
    self._step += 1
    return make_obs(self._step), 0.0, self._step >= 2, EpisodeInfo()


class DemoReplayE2ETests(unittest.TestCase):
    def test_record_then_replay_actions_through_step(self):
        record_env = _MockRoboEnv()
        session = DemoSession(record_env)
        session.reset()
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]
        session.step(Action(joint_positions=jnp.asarray([1.0, 2.0], dtype=jnp.float32)))
        session.step(Action(joint_positions=jnp.asarray([3.0, 4.0], dtype=jnp.float32)))

        replay_env = _MockRoboEnv()
        replay_env.reset()
        for action in session.iter_replay_actions():
            replay_env.step(action)
        self.assertEqual(replay_env.replayed, [[1.0, 2.0], [3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
