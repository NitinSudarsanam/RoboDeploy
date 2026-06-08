from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Observation
from robodeploy.perception.vision_predicates import ColorBlobTracker, ColorBlobTrackerTransform, count_hsv_pixels
from robodeploy.tasks.success_predicates import get_success_predicate


def _base_obs(**kwargs) -> Observation:
    defaults = dict(
        joint_positions=jnp.zeros((2,), dtype=jnp.float32),
        joint_velocities=jnp.zeros((2,), dtype=jnp.float32),
        joint_torques=jnp.zeros((2,), dtype=jnp.float32),
        ee_position=jnp.zeros((3,), dtype=jnp.float32),
        ee_orientation=jnp.asarray([1, 0, 0, 0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
    )
    defaults.update(kwargs)
    return Observation(**defaults)


class VisionPredicateTests(unittest.TestCase):
    def test_color_blob_tracker_unprojects_pose(self):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[20:28, 30:38] = (255, 0, 0)
        tracker = ColorBlobTracker(((0.0, 80.0, 80.0), (10.0, 255.0, 255.0)), min_pixels=10)
        pose = tracker.detect(
            rgb,
            None,
            {"fx": 64.0, "fy": 48.0, "cx": 32.0, "cy": 24.0},
            {"position": (0.6, 0.0, 0.5), "orientation": (1.0, 0.0, 0.0, 0.0)},
        )
        self.assertIsNotNone(pose)
        pos, _ = pose  # type: ignore[misc]
        self.assertGreater(pos[2], 0.0)

    def test_transform_populates_objects(self):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[20:28, 30:38] = (255, 0, 0)
        obs = _base_obs(
            images={"wrist_camera": rgb},
            camera_intrinsics={"wrist_camera": {"fx": 64.0, "fy": 48.0, "cx": 32.0, "cy": 24.0}},
        )
        out = ColorBlobTrackerTransform(camera="wrist_camera", object_name="source", min_pixels=10).forward(obs)
        self.assertIn("source", out.objects)

    def test_vision_target_in_view_predicate(self):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[10:20, 10:20] = (255, 0, 0)
        obs = _base_obs(rgb=rgb)
        fn = get_success_predicate("vision_target_in_view")
        self.assertTrue(
            fn(
                obs,
                target_color_hsv_range=((0.0, 80.0, 80.0), (10.0, 255.0, 255.0)),
                min_pixels=20,
            )
        )
        empty = _base_obs(rgb=np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertFalse(
            fn(
                empty,
                target_color_hsv_range=((0.0, 80.0, 80.0), (10.0, 255.0, 255.0)),
                min_pixels=20,
            )
        )

    def test_count_hsv_pixels_red_blob(self):
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[4:10, 4:10] = (255, 0, 0)
        pixels = count_hsv_pixels(rgb, lower=(0.0, 80.0, 80.0), upper=(10.0, 255.0, 255.0))
        self.assertGreaterEqual(pixels, 20)


if __name__ == "__main__":
    unittest.main()
