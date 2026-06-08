from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Observation
from robodeploy.obs_pipeline import GraspStabilityFusion, ObsPipeline


def _obs(**kwargs) -> Observation:
    defaults = dict(
        joint_positions=jnp.zeros((2,), dtype=jnp.float32),
        joint_velocities=jnp.zeros((2,), dtype=jnp.float32),
        joint_torques=jnp.zeros((2,), dtype=jnp.float32),
        ee_position=jnp.zeros((3,), dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
    )
    defaults.update(kwargs)
    return Observation(**defaults)


class GraspFusionTests(unittest.TestCase):
    def test_fusion_scores_stable_grasp_high(self):
        fusion = GraspStabilityFusion()
        obs = _obs(
            ft_force=jnp.asarray([3.0, 0.0, 0.0], dtype=jnp.float32),
            imu_angular_velocity=jnp.asarray([0.05, 0.0, 0.0], dtype=jnp.float32),
            contact_state={"wrist_contact": True},
        )
        out = fusion.forward(obs)
        score = out.metadata.get("grasp_stability")
        self.assertIsNotNone(score)
        self.assertGreater(float(score), 0.7)

    def test_fusion_scores_unstable_grasp_low(self):
        fusion = GraspStabilityFusion()
        obs = _obs(
            ft_force=jnp.asarray([0.1, 0.0, 0.0], dtype=jnp.float32),
            imu_angular_velocity=jnp.asarray([1.0, 0.5, 0.2], dtype=jnp.float32),
            contact_state={"wrist_contact": False},
        )
        out = fusion.forward(obs)
        score = float(out.metadata.get("grasp_stability", 0.0))
        self.assertLess(score, 0.3)

    def test_fusion_in_obs_pipeline(self):
        pipeline = ObsPipeline([GraspStabilityFusion()])
        obs = _obs(
            ft_force=jnp.asarray([2.5, 0.0, 0.0], dtype=jnp.float32),
            imu_angular_velocity=jnp.asarray([0.02, 0.0, 0.0], dtype=jnp.float32),
            contact_state={"wrist_contact": True},
        )
        out = pipeline.process(obs)
        self.assertIn("grasp_stability", out.metadata)


if __name__ == "__main__":
    unittest.main()
