from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class PolicyBuilderTests(unittest.TestCase):
    def test_build_reach_policy(self):
        from robodeploy.core.spaces import ActionSpace
        from robodeploy.policies.builder import PolicyBuilder
        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        policy = (
            PolicyBuilder()
            .with_action_space(ActionSpace.JOINT_POS)
            .add_carry(mode="follow")
            .add_settle_home()
            .add_reach_phase("grasp", target="source", offset=(0.0, 0.0, 0.02))
            .add_hold(steps=10)
            .build()
        )
        self.assertIsInstance(policy, ReachTrajectoryPolicy)
        self.assertEqual(policy.config["carry_mode"], "follow")

    def test_invalid_carry_raises(self):
        from robodeploy.policies.builder import PolicyBuilder

        with self.assertRaises(ValueError):
            PolicyBuilder().add_carry(mode="invalid")  # type: ignore[arg-type]
