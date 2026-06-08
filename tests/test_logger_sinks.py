from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from robodeploy.observability.logger import JsonlSink, RoboDeployLogger, StdoutSink


class LoggerSinkTests(unittest.TestCase):
    def test_jsonl_and_stdout_sinks_both_receive_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.jsonl"
            stdout = mock.Mock()
            logger = RoboDeployLogger(
                sinks=[JsonlSink(log_path), StdoutSink()],
                run_name="test-run",
            )
            with mock.patch("sys.stdout", stdout):
                logger.log_step({"reward": 1.0, "done": False})
                logger.log_episode({"success": True})
                logger.close()
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["kind"], "step")
            self.assertEqual(first["payload"]["reward"], 1.0)
            self.assertTrue(stdout.write.called)

    def test_wandb_sink_mocked(self):
        fake_wandb = mock.Mock()
        fake_run = mock.Mock()
        fake_wandb.init.return_value = fake_run
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            from robodeploy.observability.logger import WandbSink

            sink = WandbSink(project="p", run_name="r")
            sink.write(0, {"reward": 0.5}, kind="step")
            sink.close()
        fake_wandb.log.assert_called_once()
        fake_run.finish.assert_called_once()

    def test_tensorboard_sink_mocked(self):
        fake_writer = mock.Mock()
        fake_tb_module = mock.MagicMock()
        fake_tb_module.SummaryWriter = mock.Mock(return_value=fake_writer)
        fake_torch_utils = mock.MagicMock()
        fake_torch_utils.tensorboard = fake_tb_module
        fake_torch = mock.MagicMock()
        fake_torch.utils = fake_torch_utils
        with mock.patch.dict(
            "sys.modules",
            {
                "torch": fake_torch,
                "torch.utils": fake_torch_utils,
                "torch.utils.tensorboard": fake_tb_module,
            },
        ):
            from robodeploy.observability.logger import TensorBoardSink

            with tempfile.TemporaryDirectory() as tmp:
                sink = TensorBoardSink(log_dir=tmp)
                sink.write(1, {"reward": 1.25, "reward_components": {"reach": 0.5}}, kind="step")
                sink.close()
        fake_writer.add_scalar.assert_called()
        fake_writer.close.assert_called_once()

    def test_mlflow_sink_mocked(self):
        fake_mlflow = mock.Mock()
        with mock.patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            from robodeploy.observability.logger import MlflowSink

            sink = MlflowSink(experiment_name="exp", run_name="run-a")
            sink.write(2, {"reward": 2.0, "done": False}, kind="step")
            sink.close()
        fake_mlflow.set_experiment.assert_called_once_with("exp")
        fake_mlflow.start_run.assert_called_once()
        fake_mlflow.log_metrics.assert_called_once()
        fake_mlflow.end_run.assert_called_once()

    def test_robodeploy_logger_multi_sink_dispatch(self):
        fake_wandb = mock.Mock()
        fake_run = mock.Mock()
        fake_wandb.init.return_value = fake_run
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.jsonl"
            with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
                from robodeploy.observability.logger import JsonlSink, RoboDeployLogger, WandbSink

                logger = RoboDeployLogger(
                    sinks=[JsonlSink(log_path), WandbSink(project="p", run_name="r")],
                    run_name="multi",
                )
                logger.log_step({"reward": 3.0})
                logger.close()
            self.assertTrue(log_path.is_file())
            fake_wandb.log.assert_called_once()


if __name__ == "__main__":
    unittest.main()
