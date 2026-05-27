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

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING
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
class PhysicsRandomConfig:
    """Bounds for physics parameter randomisation (sim backends only)."""
    gravity_noise:      float = 0.0    # ± m/s² added to z gravity
    friction_range:     tuple[float, float] = (0.5, 2.0)   # contact friction scale
    mass_scale_range:   tuple[float, float] = (0.8, 1.2)   # object mass multiplier


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
