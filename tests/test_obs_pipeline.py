from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.types import Observation, SensorData
from robodeploy.obs_pipeline import ObsPipeline, ObsSyncMode, SensorSampleBuffer


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

    def test_sensor_buffer_merges_within_window(self):
        pipeline = ObsPipeline(sync_window_s=0.1)
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        pipeline.buffer_sensor(
            "wrist",
            SensorData(rgb=rgb, timestamp_hw=0.0, timestamp=0.0),
        )
        merged = pipeline.process(make_obs(hw=0.05))
        self.assertIn("wrist", merged.images)
        self.assertIsNotNone(merged.rgb)

    def test_sensor_buffer_merges_camera_intrinsics_and_extrinsics(self):
        pipeline = ObsPipeline(sync_window_s=0.1)
        extrinsics = {
            "position": (0.5, 0.0, 0.8),
            "orientation": (1.0, 0.0, 0.0, 0.0),
            "frame_id": "wrist_camera",
            "source": "tf",
        }
        pipeline.buffer_sensor(
            "wrist_camera",
            SensorData(
                rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                timestamp_hw=0.0,
                timestamp=0.0,
                intrinsics={"fx": 64.0, "fy": 48.0, "cx": 32.0, "cy": 24.0},
                extrinsics=extrinsics,
            ),
        )
        merged = pipeline.process(make_obs(hw=0.02))
        self.assertIn("wrist_camera", merged.camera_intrinsics)
        self.assertEqual(merged.camera_intrinsics["wrist_camera"]["fx"], 64.0)
        self.assertIn("wrist_camera", merged.camera_extrinsics)
        self.assertEqual(merged.camera_extrinsics["wrist_camera"]["source"], "tf")


if __name__ == "__main__":
    unittest.main()
