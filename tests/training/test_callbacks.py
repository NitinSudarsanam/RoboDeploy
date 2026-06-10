from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from robodeploy.demo_recording import DemoFrame
from robodeploy.training.bc import BCPolicyModule, bc_mse_loss
from robodeploy.training.callbacks import CheckpointCallback, EarlyStoppingCallback, WandbCallback
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


class CallbackTests(unittest.TestCase):
    def test_checkpoint_callback_writes_on_interval(self):
        dataset = DemoDataset([_frame(0.1) for _ in range(4)])
        module = BCPolicyModule(obs_keys=["proprio"], action_dim=2, proprio_dim=6)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpts"
            cfg = TrainerConfig(epochs=2, batch_size=4, lr=1e-3, log_dir=tmp, checkpoint_interval=10_000)
            callback = CheckpointCallback(save_dir=str(ckpt_dir), every_n_steps=1)
            trainer = Trainer(
                policy_module=module,
                dataset=dataset,
                loss_fn=bc_mse_loss,
                config=cfg,
                callbacks=[callback],
            )
            trainer.fit()
            checkpoints = list(ckpt_dir.glob("checkpoint_*.pt"))
            self.assertGreaterEqual(len(checkpoints), 1)

    def test_early_stopping_triggers_on_plateau(self):
        callback = EarlyStoppingCallback(metric="loss", patience=2, min_delta=1e-6)
        callback.on_epoch_end(None, {"loss": 1.0})
        callback.on_epoch_end(None, {"loss": 1.0})
        self.assertFalse(callback.should_stop)
        callback.on_epoch_end(None, {"loss": 1.0})
        self.assertTrue(callback.should_stop)

    def test_wandb_callback_logs_with_mocked_sink(self):
        fake_wandb = mock.Mock()
        fake_run = mock.Mock()
        fake_wandb.init.return_value = fake_run
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            callback = WandbCallback(project="robodeploy-test", config={"preset": "dummy"})
            dataset = DemoDataset([_frame(0.2) for _ in range(4)])
            module = BCPolicyModule(obs_keys=["proprio"], action_dim=2, proprio_dim=6)
            cfg = TrainerConfig(epochs=1, batch_size=4, lr=1e-3, log_dir=".", checkpoint_interval=10_000)
            trainer = Trainer(
                policy_module=module,
                dataset=dataset,
                loss_fn=bc_mse_loss,
                config=cfg,
                callbacks=[callback],
            )
            trainer.fit()
        fake_wandb.init.assert_called_once_with(
            project="robodeploy-test",
            name=None,
            config={"preset": "dummy"},
            reinit=True,
        )
        self.assertGreater(fake_wandb.log.call_count, 0)
        logged_metrics = [call.args[0] for call in fake_wandb.log.call_args_list]
        self.assertTrue(any("loss" in metrics for metrics in logged_metrics))


if __name__ == "__main__":
    unittest.main()
