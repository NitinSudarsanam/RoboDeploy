"""Multi-modal observation fusion transforms."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from robodeploy.core.transforms import ITransform
from robodeploy.core.types import Observation


class GraspStabilityFusion(ITransform):
    """Combine FT magnitude, IMU stillness, and contact into grasp_stability [0,1]."""

    def __init__(
        self,
        *,
        ft_min_N: float = 1.0,
        ft_max_N: float = 5.0,
        imu_omega_max: float = 0.3,
        contact_sensor: str = "wrist_contact",
        ft_weight: float = 0.4,
        imu_weight: float = 0.3,
        contact_weight: float = 0.3,
    ) -> None:
        self._ft_min = float(ft_min_N)
        self._ft_max = float(ft_max_N)
        self._imu_omega_max = float(imu_omega_max)
        self._contact_sensor = str(contact_sensor)
        self._ft_weight = float(ft_weight)
        self._imu_weight = float(imu_weight)
        self._contact_weight = float(contact_weight)

    def _ft_in_range(self, force) -> float:  # noqa: ANN001
        mag = float(np.linalg.norm(np.asarray(force, dtype=np.float32)))
        if mag <= self._ft_min:
            return 0.0
        if mag >= self._ft_max:
            return 1.0
        return (mag - self._ft_min) / max(self._ft_max - self._ft_min, 1e-6)

    def _imu_still(self, omega) -> float:  # noqa: ANN001
        mag = float(np.linalg.norm(np.asarray(omega, dtype=np.float32)))
        if mag >= self._imu_omega_max:
            return 0.0
        return 1.0 - mag / max(self._imu_omega_max, 1e-6)

    def forward(self, obs: Observation) -> Observation:
        ft_score = self._ft_in_range(obs.ft_force) if obs.ft_force is not None else 0.0
        imu_score = (
            self._imu_still(obs.imu_angular_velocity)
            if obs.imu_angular_velocity is not None
            else 0.0
        )
        contact_state = getattr(obs, "contact_state", None) or {}
        contact_score = 1.0 if contact_state.get(self._contact_sensor) else 0.0
        score = (
            self._ft_weight * ft_score
            + self._imu_weight * imu_score
            + self._contact_weight * contact_score
        )
        metadata = dict(getattr(obs, "metadata", {}) or {})
        metadata["grasp_stability"] = float(score)
        return replace(obs, metadata=metadata)
