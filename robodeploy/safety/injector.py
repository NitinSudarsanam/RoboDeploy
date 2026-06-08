"""Synthetic safety violations for testing the safety pipeline in simulation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from robodeploy.core.types import Observation

from .violation import Hazard, ViolationRecord, Severity


@dataclass
class _PendingInjection:
    kind: str
    remaining_steps: int
    payload: dict = field(default_factory=dict)


class SafetyViolationInjector:
    """Injects synthetic safety violations into observations for tests."""

    def __init__(self) -> None:
        self._pending: list[_PendingInjection] = []
        self._state_timeout_until: float = 0.0
        self._collision_pairs: list[tuple[str, str]] = []

    def force_spike(self, magnitude_N: float, *, duration_steps: int = 1) -> None:
        self._pending.append(
            _PendingInjection("force", max(int(duration_steps), 1), {"magnitude_N": float(magnitude_N)})
        )

    def joint_limit_excursion(self, joint_idx: int, magnitude_rad: float, *, duration_steps: int = 1) -> None:
        self._pending.append(
            _PendingInjection(
                "joint_limit",
                max(int(duration_steps), 1),
                {"joint_idx": int(joint_idx), "magnitude_rad": float(magnitude_rad)},
            )
        )

    def state_timeout(self, duration_s: float) -> None:
        self._state_timeout_until = time.time() + max(float(duration_s), 0.0)

    def collision(self, body_a: str, body_b: str) -> None:
        self._collision_pairs.append((str(body_a), str(body_b)))

    def temperature_spike(self, joint_idx: int, temp_c: float, *, duration_steps: int = 1) -> None:
        self._pending.append(
            _PendingInjection(
                "temperature",
                max(int(duration_steps), 1),
                {"joint_idx": int(joint_idx), "temp_c": float(temp_c)},
            )
        )

    def human_proximity(self, distance_m: float, *, duration_steps: int = 1) -> None:
        self._pending.append(
            _PendingInjection(
                "human_proximity",
                max(int(duration_steps), 1),
                {"distance_m": float(distance_m)},
            )
        )

    def singularity(
        self,
        joint_idx: int,
        *,
        position_rad: float,
        velocity_rad_s: float,
        duration_steps: int = 1,
    ) -> None:
        self._pending.append(
            _PendingInjection(
                "singularity",
                max(int(duration_steps), 1),
                {
                    "joint_idx": int(joint_idx),
                    "position_rad": float(position_rad),
                    "velocity_rad_s": float(velocity_rad_s),
                },
            )
        )

    def apply(self, obs: Observation) -> Observation:
        """Return a copy of ``obs`` with pending injections applied."""
        out = Observation(
            joint_positions=np.asarray(obs.joint_positions, dtype=np.float32).copy(),
            joint_velocities=np.asarray(obs.joint_velocities, dtype=np.float32).copy(),
            joint_torques=np.asarray(obs.joint_torques, dtype=np.float32).copy(),
            ee_position=np.asarray(obs.ee_position, dtype=np.float32).copy(),
            ee_orientation=np.asarray(obs.ee_orientation, dtype=np.float32).copy(),
            ee_velocity=np.asarray(obs.ee_velocity, dtype=np.float32).copy(),
            ee_angular_velocity=np.asarray(obs.ee_angular_velocity, dtype=np.float32).copy(),
            rgb=obs.rgb,
            depth=obs.depth,
            ft_force=np.asarray(obs.ft_force, dtype=np.float32).copy() if obs.ft_force is not None else None,
            ft_torque=np.asarray(obs.ft_torque, dtype=np.float32).copy() if obs.ft_torque is not None else None,
            gripper_state=obs.gripper_state,
            objects=dict(obs.objects),
            metadata=dict(obs.metadata),
            timestamp=obs.timestamp,
            timestamp_hw=obs.timestamp_hw,
            timestamp_recv=obs.timestamp_recv,
        )

        if time.time() < self._state_timeout_until:
            out.timestamp_hw = out.timestamp - 10.0

        still_pending: list[_PendingInjection] = []
        for item in self._pending:
            if item.kind == "force":
                mag = float(item.payload["magnitude_N"])
                out.ft_force = np.asarray([mag, 0.0, 0.0], dtype=np.float32)
            elif item.kind == "joint_limit":
                idx = int(item.payload["joint_idx"])
                mag = float(item.payload["magnitude_rad"])
                q = np.asarray(out.joint_positions, dtype=np.float32).copy()
                if 0 <= idx < q.shape[0]:
                    q[idx] = mag
                out.joint_positions = q
            elif item.kind == "temperature":
                pass
            elif item.kind == "human_proximity":
                dist = float(item.payload["distance_m"])
                out.metadata = dict(out.metadata)
                out.metadata["proximity_m"] = dist
            elif item.kind == "singularity":
                idx = int(item.payload["joint_idx"])
                q = np.asarray(out.joint_positions, dtype=np.float32).copy()
                if 0 <= idx < q.shape[0]:
                    q[idx] = float(item.payload["position_rad"])
                out.joint_positions = q
                dq = np.asarray(out.joint_velocities, dtype=np.float32).copy()
                if 0 <= idx < dq.shape[0]:
                    dq[idx] = float(item.payload["velocity_rad_s"])
                out.joint_velocities = dq

            item.remaining_steps -= 1
            if item.remaining_steps > 0:
                still_pending.append(item)
        self._pending = still_pending
        return out

    def synthetic_violations(self) -> list[ViolationRecord]:
        """Return violation records for injected collision pairs (test helper)."""
        out: list[ViolationRecord] = []
        for body_a, body_b in self._collision_pairs:
            out.append(
                ViolationRecord(
                    hazard=Hazard.COLLISION_IMMINENT,
                    severity=Severity.CRITICAL,
                    message=f"injected contact between {body_a!r} and {body_b!r}",
                )
            )
        self._collision_pairs.clear()
        return out
