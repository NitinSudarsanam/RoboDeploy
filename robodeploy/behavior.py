"""Simulator-neutral behavior profile (speed, stiffness, stability).

High-level presets expand into concrete values; ``backend_for_simulator`` translates
them per backend. Users override via ``behavior=`` or ``RobotDescription.default_behavior_profile()``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Literal

PresetName = Literal["default", "smooth", "fast", "demo"]
TrackingStiffness = Literal["soft", "medium", "stiff"]
PhysicsStability = Literal["safe", "balanced", "fast"]
PhysicsIntegrator = Literal["RK4", "Euler", "implicit"]


@dataclass
class BehaviorProfile:
    """Neutral knobs; ``None`` means \"derive from preset / other fields\"."""

    preset: PresetName = "default"
    tracking_stiffness: TrackingStiffness | None = None
    physics_stability: PhysicsStability | None = None

    control_hz: float | None = None
    velocity_scale: float | None = None
    kp_scale: float | None = None
    damping_scale: float | None = None
    physics_timestep: float | None = None
    physics_integrator: PhysicsIntegrator | None = None

    def merged_with(self, override: BehaviorProfile | None) -> BehaviorProfile:
        """Field-wise merge: any non-``None`` field on ``override`` wins."""
        if override is None:
            return replace(self)
        out = replace(self)
        for f in fields(override):
            val = getattr(override, f.name)
            if val is not None:
                setattr(out, f.name, val)
        return out

    def resolved(self) -> ResolvedBehaviorProfile:
        """Expand preset + optional overrides into a fully concrete profile."""
        p_str = str(self.preset)
        if p_str not in {"default", "smooth", "fast", "demo"}:
            p_str = "default"
        p: PresetName = p_str  # type: ignore[assignment]

        # Defaults per preset (before explicit self.* overrides applied via merge already in caller)
        if p == "default":
            ts: TrackingStiffness = self.tracking_stiffness or "medium"
            ps: PhysicsStability = self.physics_stability or "balanced"
            chz = self.control_hz if self.control_hz is not None else 50.0
            vs = self.velocity_scale if self.velocity_scale is not None else 0.5
            kps = self.kp_scale if self.kp_scale is not None else 1.0
            dms = self.damping_scale if self.damping_scale is not None else 1.0
        elif p == "smooth":
            ts = self.tracking_stiffness or "soft"
            ps = self.physics_stability or "balanced"
            chz = self.control_hz if self.control_hz is not None else 50.0
            vs = self.velocity_scale if self.velocity_scale is not None else 0.4
            kps = self.kp_scale if self.kp_scale is not None else 0.5
            dms = self.damping_scale if self.damping_scale is not None else 1.5
        elif p == "fast":
            ts = self.tracking_stiffness or "stiff"
            ps = self.physics_stability or "fast"
            chz = self.control_hz if self.control_hz is not None else 100.0
            vs = self.velocity_scale if self.velocity_scale is not None else 1.0
            kps = self.kp_scale if self.kp_scale is not None else 2.0
            dms = self.damping_scale if self.damping_scale is not None else 0.7
        elif p == "demo":
            ts = self.tracking_stiffness or "soft"
            ps = self.physics_stability or "safe"
            chz = self.control_hz if self.control_hz is not None else 50.0
            vs = self.velocity_scale if self.velocity_scale is not None else 0.3
            kps = self.kp_scale if self.kp_scale is not None else 0.5
            dms = self.damping_scale if self.damping_scale is not None else 1.5
        else:
            ts = self.tracking_stiffness or "medium"
            ps = self.physics_stability or "balanced"
            chz = self.control_hz if self.control_hz is not None else 50.0
            vs = self.velocity_scale if self.velocity_scale is not None else 0.5
            kps = self.kp_scale if self.kp_scale is not None else 1.0
            dms = self.damping_scale if self.damping_scale is not None else 1.0

        # Explicit overrides for tracking / stability when preset is default-like
        if self.tracking_stiffness is not None:
            ts = self.tracking_stiffness
        if self.physics_stability is not None:
            ps = self.physics_stability

        kp0, damp0 = _stiffness_table(ts)
        kp = float(kp0 * kps)
        damping = float(damp0 * dms)

        step, integr, min_mass, min_inertia = _stability_table(ps)
        timestep = float(self.physics_timestep) if self.physics_timestep is not None else step
        integrator = self.physics_integrator if self.physics_integrator is not None else integr

        return ResolvedBehaviorProfile(
            preset=p,
            tracking_stiffness=ts,
            physics_stability=ps,
            control_hz=float(chz),
            velocity_scale=float(vs),
            kp_scale=float(kps),
            damping_scale=float(dms),
            physics_timestep=timestep,
            physics_integrator=integrator,
            kp=kp,
            joint_damping=damping,
            min_mass=min_mass,
            min_inertia_diag=min_inertia,
        )


def _stiffness_table(ts: TrackingStiffness) -> tuple[float, float]:
    if ts == "soft":
        return 5.0, 2.0
    if ts == "stiff":
        return 30.0, 0.5
    return 10.0, 1.0


def _stability_table(ps: PhysicsStability) -> tuple[float, PhysicsIntegrator, float, float]:
    if ps == "safe":
        return 0.0005, "RK4", 0.05, 1e-4
    if ps == "fast":
        return 0.002, "Euler", 0.001, 1e-4
    return 0.001, "RK4", 0.01, 1e-4


@dataclass(frozen=True)
class ResolvedBehaviorProfile:
    """Concrete profile after preset expansion."""

    preset: str
    tracking_stiffness: str
    physics_stability: str
    control_hz: float
    velocity_scale: float
    kp_scale: float
    damping_scale: float
    physics_timestep: float
    physics_integrator: str
    kp: float
    joint_damping: float
    min_mass: float
    min_inertia_diag: float


__all__ = [
    "BehaviorProfile",
    "ResolvedBehaviorProfile",
    "PresetName",
    "TrackingStiffness",
    "PhysicsStability",
    "PhysicsIntegrator",
]
