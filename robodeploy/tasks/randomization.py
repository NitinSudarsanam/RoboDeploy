"""
DomainRandomizer — sim-to-real gap reduction via scene and physics variation.

Domain randomisation is one of the primary techniques for training policies
in simulation that transfer to real hardware. By exposing the policy to a
wide distribution of object poses, lighting conditions, friction values, and
sensor noise during training, the real world becomes "just another sample"
from that distribution.

DomainRandomizer is called inside TaskBase.reset_fn() each episode. It
modifies the backend's physics and scene objects before the episode begins.

Three randomisation levels:

  NONE   — deterministic: same poses every episode. Used for debugging and
            recording demonstrations.

  LIGHT  — small pose jitter only. Suitable for early training where large
            randomisation destabilises learning.

  FULL   — pose, physics, lighting, and sensor noise randomisation. Target
            setting for training policies intended for real deployment.

Extension point for Hydra:
  DomainRandomizerConfig is a plain dataclass. In the future, Hydra structured
  configs can map to it directly via OmegaConf. No changes to this file needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Iterator
import warnings

import numpy as np

if TYPE_CHECKING:
    from robodeploy.core.interfaces.backend import IBackend


class RandomLevel(Enum):
    """How much randomisation to apply each episode."""
    NONE  = auto()   # Deterministic — same scene every episode
    LIGHT = auto()   # Pose jitter only
    FULL  = auto()   # Pose + physics + lighting + sensor noise


@dataclass
class ObjectRandomConfig:
    """Randomisation bounds for a single object in the scene.

    All values in metres (position) or radians (orientation).
    """
    object_name:     str

    # Position uniform noise: sample from [center - range, center + range]
    position_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_range:  tuple[float, float, float] = (0.05, 0.05, 0.0)

    # Yaw-only rotation noise by default (safe for tabletop pick tasks)
    yaw_range:       float = 3.14159   # full rotation


@dataclass
class SensorNoiseConfig:
    """Gaussian noise stds applied via ObsPipeline at FULL randomisation level."""

    joint_pos_std: float = 0.001
    joint_vel_std: float = 0.005
    ee_pos_std: float = 0.001
    rgb_std: float = 2.0
    depth_std: float = 0.002
    ft_force_std: float = 0.05
    ft_torque_std: float = 0.01
    imu_accel_std: float = 0.02
    imu_gyro_std: float = 0.005
    seed: int | None = None


@dataclass
class PhysicsRandomConfig:
    """Bounds for physics parameter randomisation (sim backends only)."""
    gravity_noise:      float = 0.0    # ± m/s² added to z gravity
    friction_range:     tuple[float, float] = (0.5, 2.0)   # contact friction scale
    mass_scale_range:   tuple[float, float] = (0.8, 1.2)   # object mass multiplier


def resolve_random_level(name: str | RandomLevel) -> RandomLevel:
    """Parse a level name or enum value."""
    if isinstance(name, RandomLevel):
        return name
    key = str(name).strip().upper()
    if key in RandomLevel.__members__:
        return RandomLevel[key]
    raise ValueError(f"Unknown randomization level {name!r}. Expected one of: {list(RandomLevel.__members__)}")


def _parse_object_random_config(item: Any) -> ObjectRandomConfig:
    if isinstance(item, ObjectRandomConfig):
        return item
    if isinstance(item, dict):
        return ObjectRandomConfig(**item)
    raise TypeError(f"Expected ObjectRandomConfig or dict, got {type(item).__name__}")


def _parse_sensor_noise_config(item: Any) -> SensorNoiseConfig:
    if isinstance(item, SensorNoiseConfig):
        return item
    if isinstance(item, dict):
        return SensorNoiseConfig(**item)
    raise TypeError(f"Expected SensorNoiseConfig or dict, got {type(item).__name__}")


def _parse_physics_random_config(item: Any) -> PhysicsRandomConfig:
    if isinstance(item, PhysicsRandomConfig):
        return item
    if isinstance(item, dict):
        return PhysicsRandomConfig(**item)
    raise TypeError(f"Expected PhysicsRandomConfig or dict, got {type(item).__name__}")


def resolve_domain_randomizer_config(
    spec: DomainRandomizerConfig | DomainRandomizer | dict[str, Any] | None,
    *,
    default_objects: list[ObjectRandomConfig] | None = None,
) -> DomainRandomizerConfig | None:
    """Resolve YAML/dict/task config into a ``DomainRandomizerConfig``.

    Returns ``None`` when randomization is explicitly disabled (``False``).
    """
    if spec is None or spec is False:  # type: ignore[comparison-overlap]
        return None
    if isinstance(spec, DomainRandomizer):
        return spec.config
    if isinstance(spec, DomainRandomizerConfig):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"Unsupported domain_randomization spec: {type(spec).__name__}")

    level = resolve_random_level(spec.get("level", RandomLevel.LIGHT))
    objects = [
        _parse_object_random_config(item)
        for item in spec.get("objects", default_objects or [])
    ]
    physics = _parse_physics_random_config(spec.get("physics", {}))
    sensor_noise = _parse_sensor_noise_config(spec.get("sensor_noise", {}))
    return DomainRandomizerConfig(
        level=level,
        seed=spec.get("seed"),
        objects=objects,
        physics=physics,
        sensor_noise=sensor_noise,
    )


def scale_sensor_noise(
    config: DomainRandomizerConfig,
    scale: float,
) -> DomainRandomizerConfig:
    """Return a copy with all sensor noise stds multiplied by ``scale``."""
    sn = config.sensor_noise
    factor = float(scale)
    scaled = SensorNoiseConfig(
        joint_pos_std=sn.joint_pos_std * factor,
        joint_vel_std=sn.joint_vel_std * factor,
        ee_pos_std=sn.ee_pos_std * factor,
        rgb_std=sn.rgb_std * factor,
        depth_std=sn.depth_std * factor,
        ft_force_std=sn.ft_force_std * factor,
        ft_torque_std=sn.ft_torque_std * factor,
        imu_accel_std=sn.imu_accel_std * factor,
        imu_gyro_std=sn.imu_gyro_std * factor,
        seed=sn.seed,
    )
    return replace(config, sensor_noise=scaled)


def build_dr_config_from_cell(
    cell: dict[str, Any],
    *,
    base: DomainRandomizerConfig | None = None,
    default_objects: list[ObjectRandomConfig] | None = None,
) -> DomainRandomizerConfig:
    """Build a ``DomainRandomizerConfig`` from one DR sweep cell parameter dict."""
    cfg = base or DomainRandomizerConfig()
    level = resolve_random_level(cell.get("level", cfg.level))
    pos_half = float(cell.get("position_range", 0.0))
    friction = cell.get("physics_friction_range")
    if friction is None:
        friction = cfg.physics.friction_range
    else:
        friction = (float(friction[0]), float(friction[1]))
    noise_scale = float(cell.get("sensor_noise_scale", 1.0))

    objects = list(cfg.objects)
    if default_objects:
        objects = list(default_objects)
    if pos_half > 0.0 and objects:
        objects = [
            replace(
                obj,
                position_range=(
                    pos_half,
                    pos_half,
                    obj.position_range[2],
                ),
            )
            for obj in objects
        ]

    out = DomainRandomizerConfig(
        level=level,
        seed=cell.get("seed", cfg.seed),
        objects=objects,
        physics=replace(cfg.physics, friction_range=friction),
        sensor_noise=cfg.sensor_noise,
    )
    if noise_scale != 1.0:
        out = scale_sensor_noise(out, noise_scale)
    return out


def dr_config_to_dict(config: DomainRandomizerConfig) -> dict[str, Any]:
    """Serialize a config for sweep reports and YAML round-trips."""
    return {
        "level": config.level.name,
        "seed": config.seed,
        "objects": [
            {
                "object_name": o.object_name,
                "position_center": o.position_center,
                "position_range": o.position_range,
                "yaw_range": o.yaw_range,
            }
            for o in config.objects
        ],
        "physics": {
            "gravity_noise": config.physics.gravity_noise,
            "friction_range": config.physics.friction_range,
            "mass_scale_range": config.physics.mass_scale_range,
        },
        "sensor_noise": {
            "joint_pos_std": config.sensor_noise.joint_pos_std,
            "joint_vel_std": config.sensor_noise.joint_vel_std,
            "ee_pos_std": config.sensor_noise.ee_pos_std,
            "rgb_std": config.sensor_noise.rgb_std,
            "depth_std": config.sensor_noise.depth_std,
            "ft_force_std": config.sensor_noise.ft_force_std,
            "ft_torque_std": config.sensor_noise.ft_torque_std,
            "imu_accel_std": config.sensor_noise.imu_accel_std,
            "imu_gyro_std": config.sensor_noise.imu_gyro_std,
            "seed": config.sensor_noise.seed,
        },
    }


@dataclass
class DomainRandomizerConfig:
    """Full configuration for one DomainRandomizer instance.

    Extension point: this dataclass maps to a Hydra structured config.
    Add a conf/randomization/<name>.yaml that declares _target_ and fields.
    """
    level:           RandomLevel              = RandomLevel.LIGHT
    seed:            int | None               = None      # None = random seed each run
    objects:         list[ObjectRandomConfig] = field(default_factory=list)
    physics:         PhysicsRandomConfig      = field(default_factory=PhysicsRandomConfig)
    sensor_noise:    SensorNoiseConfig        = field(default_factory=SensorNoiseConfig)


class DomainRandomizer:
    """Randomises a scene each episode to reduce the sim-to-real gap.

    Typical usage inside TaskBase.reset_fn():

        def reset_fn(self, backend: IBackend) -> None:
            self.randomizer.randomize(backend)

    Args:
        config: DomainRandomizerConfig controlling what and how much to vary.
    """

    def __init__(self, config: DomainRandomizerConfig | None = None) -> None:
        self.config = config or DomainRandomizerConfig()
        self._rng   = np.random.default_rng(self.config.seed)

    def randomize(self, backend: IBackend) -> None:
        """Apply all enabled randomisations to the backend for this episode.

        Calls the appropriate randomise_* methods based on config.level.
        Safe to call on real backends: teleport_object and set_physics_params
        raise NotImplementedError for IBackend.is_real == True, so those
        calls are skipped automatically.

        Args:
            backend: The active backend whose scene will be modified.
        """
        if self.config.level == RandomLevel.NONE:
            return

        self._randomize_object_poses(backend)

        if self.config.level == RandomLevel.FULL and not backend.is_real:
            self._randomize_physics(backend)

    def obs_noise_transform(self):
        """Return a GaussianNoiseTransform for sensor/proprio noise at FULL level."""
        if self.config.level != RandomLevel.FULL:
            return None
        from robodeploy.core.transforms import GaussianNoiseTransform

        sn = self.config.sensor_noise
        return GaussianNoiseTransform(
            joint_pos_std=float(sn.joint_pos_std),
            joint_vel_std=float(sn.joint_vel_std),
            ee_pos_std=float(sn.ee_pos_std),
            rgb_std=float(sn.rgb_std),
            depth_std=float(sn.depth_std),
            ft_force_std=float(sn.ft_force_std),
            ft_torque_std=float(sn.ft_torque_std),
            imu_accel_std=float(sn.imu_accel_std),
            imu_gyro_std=float(sn.imu_gyro_std),
            seed=sn.seed if sn.seed is not None else self.config.seed,
        )

    def _randomize_object_poses(self, backend: IBackend) -> None:
        """Jitter each object's position and orientation within configured bounds.

        Args:
            backend: Backend whose teleport_object() is called per object.
        """
        for obj_cfg in self.config.objects:
            cx, cy, cz = obj_cfg.position_center
            rx, ry, rz = obj_cfg.position_range

            x = float(self._rng.uniform(cx - rx, cx + rx))
            y = float(self._rng.uniform(cy - ry, cy + ry))
            z = float(self._rng.uniform(cz - rz, cz + rz))

            try:
                backend.teleport_object(obj_cfg.object_name, (x, y, z))
            except NotImplementedError as exc:
                warnings.warn(
                    f"DomainRandomizer skipped pose randomization for '{obj_cfg.object_name}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _randomize_physics(self, backend: IBackend) -> None:
        """Vary gravity, friction, and mass for physics robustness (sim only).

        Args:
            backend: Backend whose set_physics_params() is called.
        """
        p = self.config.physics

        gravity_z = -9.81 + float(
            self._rng.uniform(-p.gravity_noise, p.gravity_noise)
        )
        friction = float(self._rng.uniform(*p.friction_range))

        try:
            backend.set_physics_params(
                gravity=[0.0, 0.0, gravity_z],
                friction=friction,
            )
        except NotImplementedError as exc:
            warnings.warn(
                f"DomainRandomizer skipped physics randomization: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def reset_seed(self, seed: int | None = None) -> None:
        """Re-seed the random number generator.

        Args:
            seed: New seed. None = random. Useful for reproducible evaluation.
        """
        self._rng = np.random.default_rng(seed)

    def seed(self, seed: int) -> None:
        """Alias for reset_seed with a required integer seed."""
        self.reset_seed(int(seed))
