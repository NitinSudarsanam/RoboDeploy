"""Action noise and external disturbance injectors for sim training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from robodeploy.core.types import Action

if TYPE_CHECKING:
    from robodeploy.core.interfaces.backend import IBackend


class ActionNoiseInjector:
    """Inject noise into actions during sim training."""

    def __init__(
        self,
        *,
        joint_noise_std: float = 0.001,
        command_dropout_p: float = 0.0,
        slip_probability: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.joint_noise_std = float(joint_noise_std)
        self.command_dropout_p = float(command_dropout_p)
        self.slip_probability = float(slip_probability)
        self._rng = np.random.default_rng(seed)
        self._last_action: Action | None = None

    def __call__(self, action: Action) -> Action:
        if action is None or action.joint_positions is None:
            return action
        q = np.asarray(action.joint_positions, dtype=np.float32).copy()
        if self.command_dropout_p > 0.0 and self._last_action is not None:
            if self._rng.random() < self.command_dropout_p:
                prev = np.asarray(self._last_action.joint_positions, dtype=np.float32)
                if prev.shape == q.shape:
                    q = prev
        if self.joint_noise_std > 0.0:
            q += self._rng.normal(0.0, self.joint_noise_std, size=q.shape).astype(np.float32)
        if self.slip_probability > 0.0 and self._rng.random() < self.slip_probability:
            q *= float(self._rng.uniform(0.5, 0.95))
        out = Action(joint_positions=q, gripper=action.gripper)
        self._last_action = out
        return out

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "ActionNoiseInjector | None":
        if not cfg:
            return None
        return cls(
            joint_noise_std=float(cfg.get("joint_noise_std", 0.001)),
            command_dropout_p=float(cfg.get("command_dropout_p", 0.0)),
            slip_probability=float(cfg.get("slip_probability", 0.0)),
            seed=cfg.get("seed"),
        )


class ExternalDisturbanceInjector:
    """Random external forces on gripper / prop during sim."""

    def __init__(
        self,
        *,
        force_range_N: tuple[float, float] = (0.0, 1.0),
        duration_steps_range: tuple[int, int] = (1, 5),
        probability_per_step: float = 0.001,
        seed: int | None = None,
    ) -> None:
        self.force_range_N = (float(force_range_N[0]), float(force_range_N[1]))
        self.duration_steps_range = (int(duration_steps_range[0]), int(duration_steps_range[1]))
        self.probability_per_step = float(probability_per_step)
        self._rng = np.random.default_rng(seed)
        self._remaining = 0
        self._active_force: np.ndarray | None = None

    def inject(self, backend: "IBackend") -> None:
        if self._remaining > 0 and self._active_force is not None:
            self._remaining -= 1
            self._apply_force(backend, self._active_force)
            if self._remaining <= 0:
                self._active_force = None
            return
        if self._rng.random() >= self.probability_per_step:
            return
        lo, hi = self.force_range_N
        force = self._rng.uniform(lo, hi, size=3).astype(np.float64)
        self._active_force = force
        self._remaining = int(self._rng.integers(self.duration_steps_range[0], self.duration_steps_range[1] + 1))
        self._apply_force(backend, force)

    def _apply_force(self, backend: "IBackend", force: np.ndarray) -> None:
        apply = getattr(backend, "apply_external_force", None)
        if callable(apply):
            try:
                apply(force)
            except Exception:
                pass

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "ExternalDisturbanceInjector | None":
        if not cfg:
            return None
        fr = cfg.get("force_range_N", (0.0, 1.0))
        dr = cfg.get("duration_steps_range", (1, 5))
        return cls(
            force_range_N=(float(fr[0]), float(fr[1])),
            duration_steps_range=(int(dr[0]), int(dr[1])),
            probability_per_step=float(cfg.get("probability_per_step", 0.001)),
            seed=cfg.get("seed"),
        )
