"""Reusable success predicate registry for manipulation tasks."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from robodeploy.core.types import Observation

SuccessFn = Callable[..., bool]

_SUCCESS_PREDICATES: dict[str, SuccessFn] = {}


def register_success(name: str):
    def _decorator(fn: SuccessFn) -> SuccessFn:
        _SUCCESS_PREDICATES[name] = fn
        return fn

    return _decorator


def get_success_predicate(name: str) -> SuccessFn:
    if name not in _SUCCESS_PREDICATES:
        raise KeyError(f"Unknown success predicate '{name}'.")
    return _SUCCESS_PREDICATES[name]


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _object_pos(obs: Observation, name: str) -> tuple[float, float, float] | None:
    objects = getattr(obs, "objects", None) or {}
    if name in objects:
        pos, _ = objects[name]
        return tuple(float(v) for v in pos)
    return None


@register_success("object_at_target")
def object_at_target(
    obs: Observation,
    *,
    source: str,
    target_pos: tuple[float, float, float],
    threshold: float = 0.04,
) -> bool:
    pos = _object_pos(obs, source)
    if pos is None:
        return False
    return _dist(pos, target_pos) < float(threshold)


@register_success("gripper_holding")
def gripper_holding(
    obs: Observation,
    *,
    source: str,
    ee_distance_max: float = 0.04,
) -> bool:
    pos = _object_pos(obs, source)
    if pos is None:
        return False
    ee = tuple(float(v) for v in obs.ee_position)
    return _dist(ee, pos) < float(ee_distance_max)


@register_success("grasp_force_min")
def grasp_force_min(obs: Observation, *, threshold_N: float = 2.0, window: int = 1) -> bool:
    """FT-based grasp confirmation."""
    ft = getattr(obs, "ft_force", None)
    if ft is None:
        forces = getattr(obs, "ft_forces", None) or {}
        if forces:
            ft = next(iter(forces.values()))
    if ft is None:
        return False
    mag = float(sum(float(v) ** 2 for v in ft) ** 0.5)
    return mag >= float(threshold_N)


@register_success("contact_held")
def contact_held(obs: Observation, *, sensor_name: str = "wrist_contact") -> bool:
    contact = getattr(obs, "contact_state", None)
    if not contact:
        return False
    return bool(contact.get(sensor_name, False))


@register_success("imu_stable")
def imu_stable(
    obs: Observation,
    *,
    max_angular_velocity: float = 0.3,
    max_acceleration: float = 2.0,
) -> bool:
    """Stability check using IMU — useful for hold-pose success."""
    omega = getattr(obs, "imu_angular_velocity", None)
    if omega is None:
        return False
    omega_norm = float(sum(float(v) ** 2 for v in omega) ** 0.5)
    accel = getattr(obs, "imu_acceleration", None)
    acc_excess = 0.0
    if accel is not None:
        acc_norm = float(sum(float(v) ** 2 for v in accel) ** 0.5)
        acc_excess = abs(acc_norm - 9.81)
    return omega_norm <= float(max_angular_velocity) and acc_excess <= float(max_acceleration)


@register_success("vision_target_in_view")
def vision_target_in_view(
    obs: Observation,
    *,
    target_color_hsv_range: tuple[tuple[float, float, float], tuple[float, float, float]],
    min_pixels: int = 100,
) -> bool:
    """Color-blob based vision target check."""
    rgb = getattr(obs, "rgb", None)
    if rgb is None and getattr(obs, "images", None):
        images = obs.images
        if images:
            rgb = next(iter(images.values()))
    if rgb is None:
        return False
    from robodeploy.perception.vision_predicates import count_hsv_pixels

    lower, upper = target_color_hsv_range
    return count_hsv_pixels(np.asarray(rgb), lower=lower, upper=upper) >= int(min_pixels)


@register_success("force_above_threshold")
def force_above_threshold(obs: Observation, *, threshold_N: float = 5.0) -> bool:
    ft = getattr(obs, "ft_force", None)
    if ft is None:
        return False
    try:
        mag = float(sum(float(v) ** 2 for v in ft) ** 0.5)
    except Exception:
        return False
    return mag >= float(threshold_N)


@register_success("grasp_force_min")
def grasp_force_min(obs: Observation, *, threshold_N: float = 2.0, window: int = 1) -> bool:
    """FT-based grasp confirmation."""
    del window  # windowing handled by policies; predicate is instantaneous.
    ft = getattr(obs, "ft_force", None)
    if ft is None and getattr(obs, "ft_forces", None):
        forces = obs.ft_forces
        if forces:
            ft = next(iter(forces.values()))
    if ft is None:
        return False
    try:
        mag = float(sum(float(v) ** 2 for v in ft) ** 0.5)
    except Exception:
        return False
    return mag >= float(threshold_N)


@register_success("contact_held")
def contact_held(obs: Observation, *, sensor_name: str = "wrist_contact") -> bool:
    contact = getattr(obs, "contact_state", None) or {}
    return bool(contact.get(sensor_name, False))


@register_success("imu_stable")
def imu_stable(
    obs: Observation,
    *,
    max_angular_velocity: float = 0.3,
    max_acceleration: float = 2.0,
) -> bool:
    """Stability check using IMU angular velocity and linear acceleration."""
    omega = getattr(obs, "imu_angular_velocity", None)
    if omega is None:
        return False
    omega_mag = float(sum(float(v) ** 2 for v in omega) ** 0.5)
    if omega_mag > float(max_angular_velocity):
        return False
    accel = getattr(obs, "imu_acceleration", None)
    if accel is None:
        return True
    accel_mag = float(sum(float(v) ** 2 for v in accel) ** 0.5)
    acc_excess = abs(accel_mag - 9.81)
    return acc_excess <= float(max_acceleration)


@register_success("vision_target_in_view")
def vision_target_in_view(
    obs: Observation,
    *,
    target_color_hsv_range: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    min_pixels: int = 100,
) -> bool:
    """Color-blob vision target check (HSV in OpenCV order)."""
    rgb = obs.rgb
    if rgb is None and obs.images:
        rgb = next(iter(obs.images.values()))
    if rgb is None:
        return False
    try:
        import cv2
    except ImportError:
        return False
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False
    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV)
    if target_color_hsv_range is None:
        low = np.array([0, 100, 100], dtype=np.uint8)
        high = np.array([10, 255, 255], dtype=np.uint8)
    else:
        low, high = target_color_hsv_range
        low = np.asarray(low, dtype=np.uint8)
        high = np.asarray(high, dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    return int(np.sum(mask > 0)) >= int(min_pixels)


@register_success("object_lifted")
def object_lifted(
    obs: Observation,
    *,
    source: str,
    initial_z: float,
    lift_min: float = 0.05,
) -> bool:
    pos = _object_pos(obs, source)
    if pos is None:
        return False
    return float(pos[2]) - float(initial_z) >= float(lift_min)


@register_success("liquid_in_target")
def liquid_in_target(
    obs: Observation,
    *,
    source: str,
    target: str,
    threshold: float = 0.06,
) -> bool:
    src = _object_pos(obs, source)
    tgt = _object_pos(obs, target)
    if src is None or tgt is None:
        return False
    return _dist(src, tgt) < float(threshold)


@register_success("peg_in_hole")
def peg_in_hole(
    obs: Observation,
    *,
    peg: str,
    hole: str,
    threshold: float = 0.03,
) -> bool:
    peg_pos = _object_pos(obs, peg)
    hole_pos = _object_pos(obs, hole)
    if peg_pos is None or hole_pos is None:
        return False
    return _dist(peg_pos, hole_pos) < float(threshold)


class CompoundSuccess:
    @classmethod
    def all_of(cls, *predicates: Callable[[Observation], bool]) -> Callable[[Observation], bool]:
        def _fn(obs: Observation) -> bool:
            return all(p(obs) for p in predicates)

        return _fn

    @classmethod
    def any_of(cls, *predicates: Callable[[Observation], bool]) -> Callable[[Observation], bool]:
        def _fn(obs: Observation) -> bool:
            return any(p(obs) for p in predicates)

        return _fn
