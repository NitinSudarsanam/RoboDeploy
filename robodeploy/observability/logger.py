"""Multi-sink structured logging for RoboDeploy runs."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable


def _generate_run_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "shape") or hasattr(value, "dtype"):
        from robodeploy.core.interop import to_numpy

        return to_numpy(value).tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@runtime_checkable
class LogSink(Protocol):
    def write(self, step: int, payload: dict, *, kind: str) -> None: ...
    def close(self) -> None: ...


class StdoutSink:
    """Human-readable step/episode lines."""

    def write(self, step: int, payload: dict, *, kind: str) -> None:
        summary = {
            k: payload[k]
            for k in ("reward", "done", "sensor_health", "episode_id")
            if k in payload
        }
        print(f"[robodeploy:{kind} step={step}] {summary}", file=sys.stdout)

    def close(self) -> None:
        return


class JsonlSink:
    """Append-only JSONL log file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    def write(self, step: int, payload: dict, *, kind: str) -> None:
        record = {
            "step": int(step),
            "kind": str(kind),
            "timestamp": time.time(),
            "payload": _jsonable(payload),
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class WandbSink:
    """Optional Weights & Biases sink (requires wandb package)."""

    def __init__(self, *, project: str | None = None, run_name: str | None = None, config: dict | None = None) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("WandbSink requires `pip install wandb`.") from exc
        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, config=config or {}, reinit=True)

    def write(self, step: int, payload: dict, *, kind: str) -> None:
        flat = {f"{kind}/{k}": v for k, v in _jsonable(payload).items() if isinstance(v, (int, float, bool, str))}
        self._wandb.log(flat, step=int(step))

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


class TensorBoardSink:
    """Optional TensorBoard scalar sink."""

    def __init__(self, log_dir: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError("TensorBoardSink requires `pip install tensorboard torch`.") from exc
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def write(self, step: int, payload: dict, *, kind: str) -> None:
        for key, value in _jsonable(payload).items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(f"{kind}/{key}", float(value), int(step))
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (int, float)):
                        self._writer.add_scalar(f"{kind}/{key}/{sub_key}", float(sub_val), int(step))

    def close(self) -> None:
        self._writer.close()


class MlflowSink:
    """Optional MLflow metric sink."""

    def __init__(self, *, experiment_name: str | None = None, run_name: str | None = None) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("MlflowSink requires `pip install mlflow`.") from exc
        self._mlflow = mlflow
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)

    def write(self, step: int, payload: dict, *, kind: str) -> None:
        metrics = {}
        for key, value in _jsonable(payload).items():
            if isinstance(value, (int, float)):
                metrics[f"{kind}.{key}"] = float(value)
        if metrics:
            self._mlflow.log_metrics(metrics, step=int(step))

    def close(self) -> None:
        self._mlflow.end_run()


class RoboDeployLogger:
    """Single entry point dispatching structured records to configured sinks."""

    def __init__(
        self,
        *,
        sinks: list[LogSink] | None = None,
        run_name: str | None = None,
        config: dict | None = None,
    ) -> None:
        self._sinks = list(sinks or [])
        self._step = 0
        self._run_name = run_name or _generate_run_name()
        self._meta = {
            "run_name": self._run_name,
            "start_time": time.time(),
            "config": config or {},
        }

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    def log_step(self, payload: dict, *, step: int | None = None) -> None:
        s = int(step) if step is not None else self._step
        for sink in self._sinks:
            sink.write(s, payload, kind="step")
        self._step = s + 1

    def log_episode(self, payload: dict) -> None:
        for sink in self._sinks:
            sink.write(self._step, payload, kind="episode")

    def log_diagnostic(
        self,
        payload: dict,
        *,
        level: Literal["info", "warn", "error"] = "info",
    ) -> None:
        body = {"level": level, **payload}
        for sink in self._sinks:
            sink.write(self._step, body, kind="diagnostic")

    def log_artifact(self, name: str, path: Path) -> None:
        payload = {"artifact": str(name), "path": str(path)}
        for sink in self._sinks:
            sink.write(self._step, payload, kind="artifact")

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()
