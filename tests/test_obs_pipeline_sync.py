from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.types import Observation, SensorData
from robodeploy.obs_pipeline import ObsPipeline, ObsSyncMode


def make_obs(*, hw: float = 0.0) -> Observation:
    return Observation(
        joint_positions=np.zeros(2, dtype=np.float32),
        joint_velocities=np.zeros(2, dtype=np.float32),
        joint_torques=np.zeros(2, dtype=np.float32),
        ee_position=np.zeros(3, dtype=np.float32),
        ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ee_velocity=np.zeros(3, dtype=np.float32),
        ee_angular_velocity=np.zeros(3, dtype=np.float32),
        timestamp=hw,
        timestamp_hw=hw,
    )


class ObsPipelineSyncTests(unittest.TestCase):
    def test_drop_latest_always_processes(self):
        pipeline = ObsPipeline(sync_mode=ObsSyncMode.DROP_LATEST, sync_window_s=0.05)
        first = pipeline.process(make_obs(hw=0.0))
        second = pipeline.process(make_obs(hw=1.0))
        self.assertIsNot(first, second)

    def test_time_window_reuses_last_processed(self):
        pipeline = ObsPipeline(sync_mode=ObsSyncMode.TIME_WINDOW, sync_window_s=0.05)
        first = pipeline.process(make_obs(hw=0.0))
        second = pipeline.process(make_obs(hw=1.0))
        self.assertIs(first, second)

    def test_reset_sync_clears_buffers(self):
        pipeline = ObsPipeline(sync_mode=ObsSyncMode.TIME_WINDOW, sync_window_s=0.05)
        pipeline.process(make_obs(hw=0.0))
        pipeline.buffer_sensor("cam", SensorData(rgb=np.zeros((2, 2, 3), dtype=np.uint8), timestamp_hw=0.0))
        pipeline.reset_sync()
        merged = pipeline.process(make_obs(hw=0.01))
        self.assertNotIn("cam", merged.images)

    def test_multi_sensor_skew_outside_window_drops_stale(self):
        pipeline = ObsPipeline(sync_window_s=0.05)
        pipeline.buffer_sensor(
            "wrist_camera",
            SensorData(rgb=np.ones((2, 2, 3), dtype=np.uint8), timestamp_hw=0.0),
        )
        pipeline.buffer_sensor(
            "wrist_ft",
            SensorData(
                ft_force=np.ones(3, dtype=np.float32),
                timestamp_hw=0.2,
            ),
        )
        merged = pipeline.process(make_obs(hw=0.01))
        self.assertIn("wrist_camera", merged.images)
        self.assertNotIn("wrist_ft", merged.ft_forces)


if __name__ == "__main__":
    unittest.main()
