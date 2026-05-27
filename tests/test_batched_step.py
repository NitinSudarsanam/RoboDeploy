from __future__ import annotations

import unittest

import numpy as np

from robodeploy.backends.base import BackendBase
from robodeploy.backends.capabilities import SupportsBatchedStep
from robodeploy.core.types import Action, Observation


def make_obs(tag: float) -> Observation:
    return Observation(
        joint_positions=np.array([tag], dtype=np.float32),
        joint_velocities=np.zeros(1, dtype=np.float32),
        joint_torques=np.zeros(1, dtype=np.float32),
        ee_position=np.zeros(3, dtype=np.float32),
        ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ee_velocity=np.zeros(3, dtype=np.float32),
        ee_angular_velocity=np.zeros(3, dtype=np.float32),
    )


class _BatchBackend(BackendBase):
    is_real = False
    control_hz = 100.0
    supported_action_spaces = []

    def _load(self, description, scene, sensors) -> None:  # noqa: ANN001
        del description, scene, sensors

    def _reset_impl(self) -> Observation:
        return make_obs(0.0)

    def _step_impl(self, action: Action) -> Observation:  # noqa: ARG002
        return make_obs(1.0)

    def _get_obs_impl(self) -> Observation:
        return make_obs(2.0)

    def _close_impl(self) -> None:
        return

    def step_multi(self, actions: list[Action]) -> list[Observation]:
        return [make_obs(float(i)) for i, _ in enumerate(actions)]

    def step_multi_batch(self, action_batches: list[list[Action]]) -> list[list[Observation]]:
        return [self.step_multi(batch) for batch in action_batches]


class BatchedStepTests(unittest.TestCase):
    def test_backend_exposes_batch_step(self):
        backend = _BatchBackend()
        backend._initialized = True
        batches = backend.step_multi_batch([[Action()], [Action(), Action()]])
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 1)
        self.assertEqual(len(batches[1]), 2)
        self.assertIsInstance(backend, SupportsBatchedStep)


if __name__ == "__main__":
    unittest.main()
