from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.types import Observation
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


class ObsPipelineTests(unittest.TestCase):
    def test_with_primary_fields_mirrors_named_images(self):
        rgb = np.ones((2, 2, 3), dtype=np.uint8)
        obs = Observation(
            joint_positions=np.zeros(2, dtype=np.float32),
            joint_velocities=np.zeros(2, dtype=np.float32),
            joint_torques=np.zeros(2, dtype=np.float32),
            ee_position=np.zeros(3, dtype=np.float32),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ee_velocity=np.zeros(3, dtype=np.float32),
            ee_angular_velocity=np.zeros(3, dtype=np.float32),
            images={"wrist": rgb},
        )
        merged = ObsPipeline.with_primary_fields(obs)
        self.assertIsNotNone(merged.rgb)
        np.testing.assert_array_equal(merged.rgb, rgb)

    def test_time_window_reuses_last_processed_observation(self):
        pipeline = ObsPipeline(sync_mode=ObsSyncMode.TIME_WINDOW, sync_window_s=0.05)
        first = pipeline.process(make_obs(hw=0.0))
        second = pipeline.process(make_obs(hw=1.0))
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
