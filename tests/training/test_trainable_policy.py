from __future__ import annotations

import unittest

import pytest

torch = pytest.importorskip("torch")

from robodeploy.core.spaces import ActionSpace
from robodeploy.policies.trainable_base import TrainablePolicyBase
from robodeploy.testing import make_obs
from robodeploy.training.bc import BCPolicyModule


class TrainablePolicyTests(unittest.TestCase):
    def test_get_action_and_batch(self):
        module = BCPolicyModule(obs_keys=["proprio"], action_dim=2, proprio_dim=6)
        policy = TrainablePolicyBase(
            module=module.to("cpu"),
            action_space=ActionSpace.JOINT_POS,
            obs_keys=["proprio"],
        )
        obs = make_obs(0.25)
        action = policy.get_action(obs)
        self.assertIsNotNone(action.joint_positions)
        batch = policy.get_action_batch([obs, make_obs(0.5)])
        self.assertEqual(len(batch), 2)

    def test_train_eval_modes(self):
        module = BCPolicyModule(obs_keys=["proprio"], action_dim=2, proprio_dim=6)
        policy = TrainablePolicyBase(
            module=module.to("cpu"),
            action_space=ActionSpace.JOINT_POS,
        )
        policy.train_mode()
        policy.eval_mode()

    def test_checkpoint_roundtrip(self):
        import tempfile
        from pathlib import Path

        tmp = Path(tempfile.mkdtemp())
        module = BCPolicyModule(obs_keys=["proprio"], action_dim=2, proprio_dim=6)
        policy = TrainablePolicyBase(
            module=module.to("cpu"),
            action_space=ActionSpace.JOINT_POS,
        )
        ckpt = tmp / "policy.pt"
        policy.save_checkpoint(ckpt)
        loaded = TrainablePolicyBase.from_checkpoint(
            ckpt,
            module=BCPolicyModule(obs_keys=["proprio"], action_dim=2, proprio_dim=6).to("cpu"),
            action_space=ActionSpace.JOINT_POS,
        )
        obs = make_obs(0.1)
        a1 = policy.get_action(obs)
        a2 = loaded.get_action(obs)
        self.assertEqual(float(a1.joint_positions[0]), float(a2.joint_positions[0]))


if __name__ == "__main__":
    unittest.main()
