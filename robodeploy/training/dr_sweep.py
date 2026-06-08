"""Domain randomization parameter sweep for sim-to-real robustness analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robodeploy.tasks.randomization import (
    DomainRandomizerConfig,
    RandomLevel,
    build_dr_config_from_cell,
    dr_config_to_dict,
)

EnvFactory = Callable[[DomainRandomizerConfig, int], Any]
PolicyFactory = Callable[[Any], Callable[[Any], Any | None]]
EpisodeRunner = Callable[[Any, Callable[[Any], Any | None], int], dict[str, float]]


def _default_episode_runner(env: Any, policy_fn: Callable[[Any], Any | None], max_steps: int) -> dict[str, float]:
    obs, info = env.reset()
    total_reward = 0.0
    success = False
    for _ in range(int(max_steps)):
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        if done:
            success = bool(getattr(info, "success", False))
            break
    return {"success": float(success), "reward": total_reward}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(np.asarray(x, dtype=np.float64))
    ry = _rankdata(np.asarray(y, dtype=np.float64))
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def iter_dr_sweep_cells(config: DRSweepConfig) -> list[dict[str, Any]]:
    """Enumerate the Cartesian product of sweep axes."""
    cells: list[dict[str, Any]] = []
    for level in config.levels:
        for pos_half in config.object_position_ranges:
            half = float(pos_half[0]) if isinstance(pos_half, (tuple, list)) else float(pos_half)
            for friction in config.physics_friction_ranges:
                lo, hi = float(friction[0]), float(friction[1])
                for noise_scale in config.sensor_noise_scales:
                    cells.append(
                        {
                            "level": level,
                            "position_range": half,
                            "physics_friction_range": (lo, hi),
                            "sensor_noise_scale": float(noise_scale),
                        }
                    )
    return cells


@dataclass
class DRSweepConfig:
    """Axes for a domain-randomization sensitivity sweep."""

    n_seeds: int = 3
    n_episodes_per_seed: int = 5
    max_steps_per_episode: int = 100
    levels: list[RandomLevel] = field(
        default_factory=lambda: [RandomLevel.NONE, RandomLevel.LIGHT, RandomLevel.FULL]
    )
    object_position_ranges: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (0.02, 0.02), (0.05, 0.05)]
    )
    physics_friction_ranges: list[tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 1.0), (0.7, 1.3), (0.5, 1.5)]
    )
    sensor_noise_scales: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])


@dataclass
class DRSweepReport:
    """Aggregated sweep results and sensitivity summary."""

    cells: list[dict[str, Any]] = field(default_factory=list)
    sensitivity: dict[str, float] = field(default_factory=dict)
    robust_params: dict[str, Any] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        return {
            "cells": self.cells,
            "sensitivity": self.sensitivity,
            "robust_params": self.robust_params,
        }

    def to_json(self) -> str:
        return json.dumps(self.report(), indent=2)

    def write_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    def plot_heatmap(
        self,
        x_param: str,
        y_param: str,
        *,
        metric: str = "success_rate",
        out_path: str | Path,
    ) -> Path:
        """Write a 2D heatmap PNG (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_heatmap") from exc

        xs = sorted({cell["params"].get(x_param) for cell in self.cells})
        ys = sorted({cell["params"].get(y_param) for cell in self.cells}, reverse=True)
        grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
        for cell in self.cells:
            params = cell["params"]
            xi = xs.index(params.get(x_param))
            yi = ys.index(params.get(y_param))
            grid[yi, xi] = float(cell.get(metric, np.nan))

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(grid, aspect="auto", origin="upper")
        ax.set_xticks(range(len(xs)), labels=[str(v) for v in xs])
        ax.set_yticks(range(len(ys)), labels=[str(v) for v in ys])
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f"{metric} vs {x_param} / {y_param}")
        fig.colorbar(im, ax=ax)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out


class DRSweep:
    """Evaluate a policy under a grid of DR settings and rank robustness."""

    def __init__(
        self,
        *,
        env_fn: EnvFactory,
        policy_fn: PolicyFactory | Callable[[Any], Callable[[Any], Any | None]],
        config: DRSweepConfig | None = None,
        episode_runner: EpisodeRunner | None = None,
        base_dr_config: DomainRandomizerConfig | None = None,
    ) -> None:
        self._env_fn = env_fn
        if not callable(policy_fn):
            raise TypeError("policy_fn must be callable")
        self._policy_fn = policy_fn
        self.config = config or DRSweepConfig()
        self._episode_runner = episode_runner or _default_episode_runner
        self._base_dr = base_dr_config
        self._report = DRSweepReport()

    @property
    def report(self) -> DRSweepReport:
        return self._report

    def run(self) -> DRSweepReport:
        cells_out: list[dict[str, Any]] = []
        for cell_params in iter_dr_sweep_cells(self.config):
            dr_cfg = build_dr_config_from_cell(cell_params, base=self._base_dr)
            seed_results: list[dict[str, float]] = []
            for seed_idx in range(self.config.n_seeds):
                seed = int(seed_idx)
                dr_cfg_seed = DomainRandomizerConfig(
                    level=dr_cfg.level,
                    seed=seed,
                    objects=dr_cfg.objects,
                    physics=dr_cfg.physics,
                    sensor_noise=dr_cfg.sensor_noise,
                )
                for _ in range(self.config.n_episodes_per_seed):
                    env = self._env_fn(dr_cfg_seed, seed)
                    try:
                        policy = self._policy_fn(env)
                        result = self._episode_runner(
                            env,
                            policy,
                            self.config.max_steps_per_episode,
                        )
                        seed_results.append(result)
                    finally:
                        close = getattr(env, "close", None)
                        if callable(close):
                            close()

            successes = [r["success"] for r in seed_results]
            rewards = [r["reward"] for r in seed_results]
            serialized_params = {
                "level": cell_params["level"].name,
                "position_range": cell_params["position_range"],
                "physics_friction_range": cell_params["physics_friction_range"],
                "sensor_noise_scale": cell_params["sensor_noise_scale"],
            }
            cells_out.append(
                {
                    "params": serialized_params,
                    "dr_config": dr_config_to_dict(dr_cfg),
                    "success_rate": float(np.mean(successes)) if successes else 0.0,
                    "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                    "std_reward": float(np.std(rewards)) if rewards else 0.0,
                    "n_episodes": len(seed_results),
                }
            )

        sensitivity = self._compute_sensitivity(cells_out)
        robust = max(cells_out, key=lambda c: (c["success_rate"], c["mean_reward"]), default={})
        self._report = DRSweepReport(
            cells=cells_out,
            sensitivity=sensitivity,
            robust_params=robust.get("params", {}),
        )
        return self._report

    def _compute_sensitivity(self, cells: list[dict[str, Any]]) -> dict[str, float]:
        if not cells:
            return {}
        success = np.asarray([c["success_rate"] for c in cells], dtype=np.float64)
        out: dict[str, float] = {}
        for key in ("position_range", "sensor_noise_scale"):
            xs = np.asarray([c["params"][key] for c in cells], dtype=np.float64)
            corr = _spearman_corr(xs, success)
            if not np.isnan(corr):
                out[key] = corr
        level_map = {name: idx for idx, name in enumerate(RandomLevel.__members__)}
        level_vals = np.asarray(
            [level_map.get(str(c["params"]["level"]), 0) for c in cells],
            dtype=np.float64,
        )
        corr = _spearman_corr(level_vals, success)
        if not np.isnan(corr):
            out["level"] = corr
        return out
