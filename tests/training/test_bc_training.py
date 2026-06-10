from __future__ import annotations

import unittest

import pytest

torch = pytest.importorskip("torch")

from robodeploy.demo_recording import DemoFrame
from robodeploy.training.bc import BCPolicyModule, bc_mse_loss, train_bc
from robodeploy.training.dataset import DemoDataset
from robodeploy.training.trainer import Trainer, TrainerConfig


def _frame(value: float) -> DemoFrame:
    return DemoFrame(
        observation={
            "joint_positions": [value, value * 0.5],
            "joint_velocities": [0.0, 0.0],
            "joint_torques": [0.0, 0.0],
        },
        action={"joint_positions": [value + 0.2, value + 0.1]},
        reward=0.0,
        done=False,
    )


class BCTrainingTests(unittest.TestCase):
    def test_overfit_small_dataset(self):
        frames = [_frame(float(i) * 0.1) for i in range(10)]
        dataset = DemoDataset(frames)
        cfg = TrainerConfig(epochs=500, batch_size=10, lr=1e-3, log_dir=".", checkpoint_interval=10_000)
        module = BCPolicyModule(obs_keys=["proprio"], action_dim=2, proprio_dim=6)
        trainer = Trainer(
            policy_module=module,
            dataset=dataset,
            loss_fn=bc_mse_loss,
            config=cfg,
        )
        metrics = trainer.fit()
        self.assertLess(metrics["loss"], 1e-4)

    def test_train_bc_convenience(self):
        dataset = DemoDataset([_frame(0.3) for _ in range(8)])
        cfg = TrainerConfig(epochs=20, batch_size=8, lr=5e-3, log_dir=".", checkpoint_interval=10_000)
        module = train_bc(dataset=dataset, config=cfg)
        batch = dataset[0]
        with torch.no_grad():
            pred = module({"proprio": batch["obs"]["proprio"].unsqueeze(0)})
        loss = float(bc_mse_loss(pred, batch["action"].unsqueeze(0)).item())
        self.assertLess(loss, 1e-2)


if __name__ == "__main__":
    unittest.main()
