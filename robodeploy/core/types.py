"""
Core types and data structures for RoboDeploy.
All values use SI units: Meters, Radians, Seconds, Newtons.
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp


@dataclass
class Observation:
    """
    Standard observation from a robot.
    
    All spatial values in SI units (meters, radians, seconds).
    All arrays are JAX arrays (jnp.ndarray) for zero-copy compatibility.
    """

    # Proprioceptive sensors
    joint_positions: jnp.ndarray  # [n_joints] in radians
    joint_velocities: jnp.ndarray  # [n_joints] in rad/s
    joint_torques: jnp.ndarray  # [n_joints] in N⋅m

    # End-effector state
    ee_position: jnp.ndarray  # [3] in meters (x, y, z)
    ee_orientation: jnp.ndarray  # [4] quaternion (w, xi, xj, xk)
    ee_velocity: jnp.ndarray  # [3] in m/s
    ee_angular_velocity: jnp.ndarray  # [3] in rad/s

    # Vision
    rgb: Optional[jnp.ndarray] = None  # [H, W, 3] uint8 RGB image
    depth: Optional[jnp.ndarray] = None  # [H, W] float32 depth in meters

    # Inertial sensors
    imu_acceleration: Optional[jnp.ndarray] = None  # [3] in m/s²
    imu_angular_velocity: Optional[jnp.ndarray] = None  # [3] in rad/s

    # Force/Torque sensors (if available)
    ft_force: Optional[jnp.ndarray] = None  # [3] in Newtons
    ft_torque: Optional[jnp.ndarray] = None  # [3] in N⋅m

    # Timestamp
    timestamp: float = 0.0  # in seconds


@dataclass
class Action:
    """
    Standard action command to a robot.
    
    All values in SI units.
    """

    # Joint-level commands (choose one based on mode)
    joint_positions: Optional[jnp.ndarray] = None  # [n_joints] in radians
    joint_velocities: Optional[jnp.ndarray] = None  # [n_joints] in rad/s
    joint_torques: Optional[jnp.ndarray] = None  # [n_joints] in N⋅m

    # Task-space commands (alternative)
    ee_position: Optional[jnp.ndarray] = None  # [3] in meters
    ee_orientation: Optional[jnp.ndarray] = None  # [4] quaternion
    ee_velocity: Optional[jnp.ndarray] = None  # [3] in m/s

    # Gripper command (0.0 = open, 1.0 = closed)
    gripper: Optional[float] = None

    # Timestamp
    timestamp: float = 0.0  # in seconds
