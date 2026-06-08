from __future__ import annotations

import time
import unittest

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.learned.diffusion import DiffusionPolicy
from robodeploy.policies.remote.server import StreamingPolicyServer


def _obs() -> Observation:
    return Observation(
        joint_positions=np.zeros(7, dtype=np.float32),
        joint_velocities=np.zeros(7, dtype=np.float32),
        joint_torques=np.zeros(7, dtype=np.float32),
        ee_position=np.zeros(3, dtype=np.float32),
        ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ee_velocity=np.zeros(3, dtype=np.float32),
        ee_angular_velocity=np.zeros(3, dtype=np.float32),
        language_instruction="reach forward",
    )


class _MockTransport:
    def connect(self) -> None:
        return

    def close(self) -> None:
        return


class StreamingPolicyServerTests(unittest.TestCase):
    def test_infer_stream_yields_first_chunk_quickly(self):
        policy = DiffusionPolicy(config={"plan_horizon": 8, "replan_interval": 8})
        server = StreamingPolicyServer(policy=policy, transport=_MockTransport(), chunk_size=2, verbose=False)
        t0 = time.perf_counter()
        chunks = server.infer_stream(_obs())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertGreaterEqual(len(chunks), 1)
        self.assertLess(elapsed_ms, 50.0)
        self.assertIsInstance(chunks[0], Action)


if __name__ == "__main__":
    unittest.main()
