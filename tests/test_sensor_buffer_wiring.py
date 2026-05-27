from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.types import Observation, SensorData
from robodeploy.testing import DummyBackend, make_obs


class _FakeSensor:
    name = "cam"

    def read(self) -> SensorData:
        return SensorData(rgb=np.zeros((2, 2, 3), dtype=np.uint8), timestamp_hw=0.0)


class SensorBufferWiringTests(unittest.TestCase):
    def test_merge_then_drain_sensor_reads(self):
        backend = DummyBackend()
        backend._sensors = [_FakeSensor()]
        merged = backend._merge_sensor_data(make_obs(0.0), backend._sensors)
        self.assertIsNotNone(merged)
        drained = backend.drain_sensor_reads()
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0][0], "cam")
        self.assertEqual(backend.drain_sensor_reads(), [])

    def test_depth_buffered_into_obs_pipeline(self):
        from robodeploy.obs_pipeline import ObsPipeline

        pipeline = ObsPipeline(sync_window_s=0.1)
        pipeline.buffer_sensor(
            "wrist",
            SensorData(depth=np.ones((4, 4), dtype=np.float32), timestamp_hw=0.0),
        )
        merged = pipeline.process(make_obs(0.0))
        self.assertIn("wrist", merged.depths)
        self.assertIsNotNone(merged.depth)


if __name__ == "__main__":
    unittest.main()
