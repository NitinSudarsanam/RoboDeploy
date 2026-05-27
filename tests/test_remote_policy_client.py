from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Observation
from robodeploy.policies.learned.diffusion import DiffusionPolicy
from robodeploy.policies.learned.vla import VLAPolicy
from robodeploy.policies.remote.http_client import HttpRemotePolicyClient, to_jsonable


def make_obs(*, instruction: str | None = None) -> Observation:
    return Observation(
        joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_angular_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        rgb=np.ones((2, 2, 3), dtype=np.uint8),
        language_instruction=instruction,
    )


class RemotePolicyClientTests(unittest.TestCase):
    def test_to_jsonable_serializes_observation_and_arrays(self):
        payload = to_jsonable({"obs": make_obs(instruction="hello")})
        self.assertEqual(payload["obs"]["language_instruction"], "hello")
        self.assertEqual(payload["obs"]["rgb"][0][0], [1, 1, 1])

    def test_vla_uses_remote_transport_when_configured(self):
        seen = {}

        def transport(endpoint: str, payload: dict):
            seen["endpoint"] = endpoint
            seen["payload"] = payload
            return {"ee_position": [0.2, 0.0, 0.0], "gripper": 1.0}

        policy = VLAPolicy(
            config={
                "action_space": "delta_ee",
                "remote_url": "http://example/vla",
                "remote_transport": transport,
            }
        )
        action = policy.get_action(make_obs(instruction="pick"))

        self.assertEqual(seen["endpoint"], "http://example/vla")
        self.assertEqual(seen["payload"]["inputs"]["instruction"], "pick")
        self.assertAlmostEqual(float(action.ee_position[0]), 0.2)
        self.assertEqual(action.gripper, 1.0)

    def test_diffusion_uses_remote_transport_for_plan_and_batch(self):
        seen = {"single": 0, "batch": 0}

        def transport(endpoint: str, payload: dict):
            if isinstance(payload["inputs"], list):
                seen["batch"] += 1
                return [
                    [{"joint_positions": [1.0, 2.0]}],
                    [{"joint_positions": [3.0, 4.0]}],
                ]
            seen["single"] += 1
            return [{"joint_positions": [5.0, 6.0]}]

        policy = DiffusionPolicy(
            config={
                "action_space": "joint_pos",
                "remote_url": "http://example/diffusion",
                "remote_transport": transport,
            }
        )
        single = policy.get_action(make_obs(instruction="single"))
        batch = policy.get_action_batch([make_obs(instruction="a"), make_obs(instruction="b")])

        self.assertEqual(seen["single"], 1)
        self.assertEqual(seen["batch"], 1)
        self.assertAlmostEqual(float(single.joint_positions[0]), 5.0)
        self.assertAlmostEqual(float(batch[0].joint_positions[0]), 1.0)
        self.assertAlmostEqual(float(batch[1].joint_positions[0]), 3.0)


if __name__ == "__main__":
    unittest.main()
