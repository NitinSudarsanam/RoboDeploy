"""Trainer callbacks for logging, checkpointing, and evaluation."""

from __future__ import annotations

from pathlib import Path

from robodeploy.training.trainer import Trainer, TrainerCallback


class CheckpointCallback(TrainerCallback):
    def __init__(
        self,
        *,
        save_dir: str,
        every_n_steps: int = 1000,
        keep_top_k: int = 5,
        metric: str = "eval/success_rate",
        mode: str = "max",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.every_n_steps = int(every_n_steps)
        self.keep_top_k = int(keep_top_k)
        self.metric = metric
        self.mode = mode
        self._best: list[tuple[float, Path]] = []
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        if trainer.global_step % self.every_n_steps != 0:
            return
        path = self.save_dir / f"checkpoint_{trainer.global_step}.pt"
        trainer.save_checkpoint(str(path))
        self.on_checkpoint(trainer, str(path))

    def on_checkpoint(self, trainer: Trainer, path: str) -> None:
        del trainer
        score = 0.0
        if self._best:
            score = self._best[0][0]
        p = Path(path)
        self._best.append((score, p))
        self._best.sort(key=lambda item: item[0], reverse=(self.mode == "max"))
        while len(self._best) > self.keep_top_k:
            _, old = self._best.pop()
            if old.exists() and old != p:
                old.unlink(missing_ok=True)


class EvalCallback(TrainerCallback):
    def __init__(self, *, eval_env, n_episodes: int = 10, deterministic: bool = True) -> None:
        self.eval_env = eval_env
        self.n_episodes = int(n_episodes)
        self.deterministic = bool(deterministic)
        self.last_metrics: dict[str, float] = {}

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        del metrics
        if self.eval_env is None:
            return
        trainer.eval_env = self.eval_env
        self.last_metrics = trainer.evaluate(n_episodes=self.n_episodes)


class WandbCallback(TrainerCallback):
    """Log training metrics to Weights & Biases (optional ``wandb`` package)."""

    def __init__(
        self,
        *,
        project: str = "robodeploy",
        run_name: str | None = None,
        config: dict | None = None,
        log_interval: int = 1,
    ) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.log_interval = max(1, int(log_interval))
        self._run = None

    def on_train_begin(self, trainer: Trainer) -> None:
        try:
            import wandb
        except ImportError:
            return
        self._run = wandb.init(
            project=self.project,
            name=self.run_name,
            config=self.config,
            reinit=True,
        )

    def on_step_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        if self._run is None or trainer.global_step % self.log_interval != 0:
            return
        import wandb

        wandb.log(metrics, step=trainer.global_step)

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        del trainer
        if self._run is None:
            return
        import wandb

        wandb.log(metrics)

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        if self._run is None:
            return
        import wandb

        wandb.log({f"epoch/{k}": v for k, v in metrics.items()}, step=trainer.global_step)


class TensorBoardCallback(TrainerCallback):
    def __init__(self, log_dir: str) -> None:
        self.log_dir = str(log_dir)
        self._writer = None

    def on_train_begin(self, trainer: Trainer) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return
        self._writer = SummaryWriter(self.log_dir)

    def on_step_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        if self._writer is None:
            return
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, trainer.global_step)

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        del trainer
        if self._writer is None:
            return
        for key, value in metrics.items():
            self._writer.add_scalar(key, value)

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        if self._writer is None:
            return
        for key, value in metrics.items():
            self._writer.add_scalar(f"epoch/{key}", value, int(metrics.get("epoch", 0)))


class DeployableCheckpointCallback(TrainerCallback):
    """Write RoboDeploy ModelSpec sidecars next to training checkpoints (GOAL 09 hook)."""

    def __init__(
        self,
        *,
        framework: str = "custom",
        expected_action_dim: int,
        expected_obs_keys: list[str] | None = None,
    ) -> None:
        from robodeploy.core.spaces import ActionSpace
        from robodeploy.policies.learned.hooks import DeployableCheckpointHook

        self._hook = DeployableCheckpointHook()
        self._spec = {
            "framework": framework,
            "expected_action_space": ActionSpace.JOINT_POS,
            "expected_action_dim": int(expected_action_dim),
            "expected_obs_keys": list(expected_obs_keys or ["proprio"]),
        }

    def on_checkpoint(self, trainer: Trainer, path: str) -> None:
        del trainer
        spec = dict(self._spec)
        spec["checkpoint"] = str(path)
        self._hook.on_checkpoint_saved(Path(path), spec)  # type: ignore[arg-type]


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, *, metric: str = "loss", patience: int = 10, min_delta: float = 1e-4) -> None:
        self.metric = metric
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self._best = float("inf")
        self._wait = 0
        self.should_stop = False

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        del trainer
        value = float(metrics.get(self.metric, float("inf")))
        if value + self.min_delta < self._best:
            self._best = value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.should_stop = True
