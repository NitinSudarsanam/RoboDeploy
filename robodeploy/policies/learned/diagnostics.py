"""Online policy action diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from robodeploy.core.types import Action


def _action_vector(action: Action) -> np.ndarray | None:
    for field_name in ("joint_positions", "joint_velocities", "ee_position"):
        value = getattr(action, field_name, None)
        if value is not None:
            return np.asarray(value, dtype=np.float64).reshape(-1)
    return None


@dataclass
class PolicyDiagnostics:
    """Track action distribution online and flag suspicious behavior."""

    _values: list[np.ndarray] = field(default_factory=list)
    nan_count: int = 0
    clipped_count: int = 0
    dim_mismatch_count: int = 0
    expected_dim: int | None = None

    def record(self, action: Action) -> None:
        vector = _action_vector(action)
        if vector is None:
            self.dim_mismatch_count += 1
            return
        if self.expected_dim is not None and vector.size != self.expected_dim:
            self.dim_mismatch_count += 1
        if not np.all(np.isfinite(vector)):
            self.nan_count += 1
            return
        self._values.append(vector.copy())

    def summary(self) -> dict:
        if not self._values:
            return {
                "count": 0,
                "action_mean": [],
                "action_std": [],
                "action_min": [],
                "action_max": [],
                "nan_count": self.nan_count,
                "clipped_count": self.clipped_count,
                "dim_mismatch_count": self.dim_mismatch_count,
            }
        stacked = np.stack(self._values, axis=0)
        return {
            "count": int(stacked.shape[0]),
            "action_mean": stacked.mean(axis=0).tolist(),
            "action_std": stacked.std(axis=0).tolist(),
            "action_min": stacked.min(axis=0).tolist(),
            "action_max": stacked.max(axis=0).tolist(),
            "nan_count": self.nan_count,
            "clipped_count": self.clipped_count,
            "dim_mismatch_count": self.dim_mismatch_count,
        }

    def reset(self) -> None:
        self._values.clear()
        self.nan_count = 0
        self.clipped_count = 0
        self.dim_mismatch_count = 0
