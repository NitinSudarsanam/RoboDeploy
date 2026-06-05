from __future__ import annotations

import unittest

import numpy as np

from examples.perception.color_blob import ColorBlobCentroidTransform
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


if __name__ == "__main__":
    unittest.main()
