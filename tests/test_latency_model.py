from __future__ import annotations

import unittest

import numpy as np

from robodeploy.bridge import LatencyModel
from robodeploy.core.types import Action


class LatencyModelTests(unittest.TestCase):
    def test_interpolate_between_buffered_actions(self):
        model = LatencyModel(mean_delay_s=0.02, jitter_std_s=0.0, seed=0)
        buf = [
            (0.0, Action(joint_positions=np.array([0.0], dtype=np.float32))),
            (0.1, Action(joint_positions=np.array([1.0], dtype=np.float32))),
        ]
        act = model.interpolate_command(buf, now=0.06)
        assert act is not None
        self.assertAlmostEqual(float(act.joint_positions[0]), 0.4, places=1)

    def test_predict_execution_time_adds_delay(self):
        model = LatencyModel(mean_delay_s=0.05, jitter_std_s=0.0, seed=0)
        t = model.predict_execution_time(1.0)
        self.assertAlmostEqual(t, 1.05, places=3)


if __name__ == "__main__":
    unittest.main()
