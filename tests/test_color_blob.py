from __future__ import annotations

import unittest

import numpy as np

from examples.perception.color_blob import ColorBlobCentroidTransform, _camera_to_world, _quat_rotate_wxyz
from robodeploy.core.types import Observation

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]


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


class ColorBlobTransformTests(unittest.TestCase):
    def test_red_blob_populates_objects(self):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[20:28, 30:38] = (255, 0, 0)
        obs = _base_obs(
            images={"wrist_camera": rgb},
            camera_intrinsics={"wrist_camera": {"fx": 64.0, "fy": 48.0, "cx": 32.0, "cy": 24.0}},
        )
        out = ColorBlobCentroidTransform(camera="wrist_camera", object_name="source").forward(obs)
        self.assertIn("source", out.objects)
        pos, _ = out.objects["source"]
        self.assertGreater(pos[2], 0.0)

    def test_quat_rotate_identity(self):
        out = _quat_rotate_wxyz((1.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.3))
        self.assertAlmostEqual(out[0], 0.1, places=5)
        self.assertAlmostEqual(out[1], 0.2, places=5)
        self.assertAlmostEqual(out[2], 0.3, places=5)

    def test_camera_to_world_with_extrinsics(self):
        pos = _camera_to_world(
            (0.1, 0.0, 0.5),
            {"position": (1.0, 0.0, 0.0), "orientation": (1.0, 0.0, 0.0, 0.0)},
            fallback_origin=(0.0, 0.0, 0.0),
            fallback_scale=(1.0, 1.0, 1.0),
            default_z=0.38,
        )
        self.assertAlmostEqual(pos[0], 1.1, places=5)

    def test_uses_depth_map_for_z_when_present(self):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[20:28, 30:38] = (255, 0, 0)
        depth = np.full((48, 64), 0.55, dtype=np.float32)
        obs = _base_obs(
            images={"wrist_camera": rgb},
            depths={"wrist_camera": depth},
            camera_intrinsics={"wrist_camera": {"fx": 64.0, "fy": 48.0, "cx": 32.0, "cy": 24.0}},
            camera_extrinsics={
                "wrist_camera": {
                    "position": (0.0, 0.0, 0.0),
                    "orientation": (1.0, 0.0, 0.0, 0.0),
                }
            },
        )
        out = ColorBlobCentroidTransform(
            camera="wrist_camera",
            object_name="source",
            default_z=0.38,
        ).forward(obs)
        pos, _ = out.objects["source"]
        self.assertAlmostEqual(pos[2], 0.55, delta=0.05)

    def test_camera_to_world_with_oriented_extrinsics(self):
        pos = _camera_to_world(
            (0.0, 0.0, 0.1),
            {
                "position": (0.0, 0.0, 0.0),
                "orientation": (0.70710678, 0.70710678, 0.0, 0.0),
            },
            fallback_origin=(0.0, 0.0, 0.0),
            fallback_scale=(1.0, 1.0, 1.0),
            default_z=0.38,
        )
        self.assertAlmostEqual(abs(pos[1]), 0.1, places=4)
        self.assertAlmostEqual(pos[0], 0.0, places=4)

    def test_uses_camera_extrinsics_mount_position(self):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[20:28, 30:38] = (255, 0, 0)
        obs = _base_obs(
            images={"wrist_camera": rgb},
            camera_intrinsics={"wrist_camera": {"fx": 64.0, "fy": 48.0, "cx": 32.0, "cy": 24.0}},
            camera_extrinsics={"wrist_camera": {"parent_link": "ee_link", "position": (0.6, 0.1, 0.5)}},
        )
        out = ColorBlobCentroidTransform(camera="wrist_camera", object_name="source").forward(obs)
        pos, _ = out.objects["source"]
        self.assertAlmostEqual(pos[0], 0.6, delta=0.2)


if __name__ == "__main__":
    unittest.main()
