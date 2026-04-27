"""
Core types and data structures for RoboDeploy.

All spatial values use SI units: metres, radians, seconds, newtons.
All arrays are JAX arrays (jnp.ndarray) for zero-copy sim compatibility.
Real backends convert their numpy/torch outputs to JAX before returning.

These types are the ONLY shared data contract between backends, policies,
tasks, and sensors. Nothing outside core/ should define its own observation
or action format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from robodeploy.core.spaces import AssetFormat

try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Primary data contract: Observation and Action
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """Full robot observation returned by every backend and sensor pipeline.

    Fields marked Optional are populated only when the task's ObsSpec
    requests them and the backend/sensor supports them.

    Sim backends populate all fields exactly.
    Real backends populate what hardware provides; missing fields are None.
    ObsPipeline normalises both sides to the same distribution before the
    policy sees the data.
    """

    # --- Proprioception (always populated) ----------------------------------
    joint_positions:      jnp.ndarray          # [dof]     radians
    joint_velocities:     jnp.ndarray          # [dof]     rad/s
    joint_torques:        jnp.ndarray          # [dof]     N·m

    # --- End-effector (always populated) ------------------------------------
    ee_position:          jnp.ndarray          # [3]       metres
    ee_orientation:       jnp.ndarray          # [4]       quaternion (w, x, y, z)
    ee_velocity:          jnp.ndarray          # [3]       m/s
    ee_angular_velocity:  jnp.ndarray          # [3]       rad/s

    # --- Vision (populated when ObsSpec.rgb / .depth is True) ---------------
    rgb:                  Optional[jnp.ndarray] = None   # [H, W, 3]  uint8
    depth:                Optional[jnp.ndarray] = None   # [H, W]     float32 metres

    # --- Force / torque sensor (populated when hardware present) -----------
    ft_force:             Optional[jnp.ndarray] = None   # [3]  Newtons
    ft_torque:            Optional[jnp.ndarray] = None   # [3]  N·m

    # --- IMU (populated when hardware present) ------------------------------
    imu_acceleration:     Optional[jnp.ndarray] = None   # [3]  m/s²
    imu_angular_velocity: Optional[jnp.ndarray] = None   # [3]  rad/s

    # --- Gripper (0.0 = fully open, 1.0 = fully closed) --------------------
    gripper_state:        Optional[float] = None

    # --- Metadata -----------------------------------------------------------
    # timestamp:      anchor time this observation represents (sim time or wall clock)
    # timestamp_hw:   hardware-level timestamp from the sensor/controller itself.
    #                 On sim backends, equals timestamp. On real backends, use the
    #                 controller's own clock (e.g. FRI timestamp for Franka).
    #                 Critical for compensating sensor latency in dynamic tasks.
    # timestamp_recv: host machine time when this observation arrived.
    #                 (timestamp_recv - timestamp_hw) gives the pipeline latency.
    timestamp:            float = 0.0
    timestamp_hw:         float = 0.0
    timestamp_recv:       float = 0.0


@dataclass
class Action:
    """Control command produced by a policy and consumed by a backend.

    Populate exactly the fields that match the policy's declared ActionSpace.
    The SafetyFilter validates and clamps these values before the backend
    receives them — policies should not self-clamp.
    """

    # Joint-space commands (ActionSpace.JOINT_POS / JOINT_VEL / JOINT_TORQUE)
    joint_positions:  Optional[jnp.ndarray] = None   # [dof]  radians
    joint_velocities: Optional[jnp.ndarray] = None   # [dof]  rad/s
    joint_torques:    Optional[jnp.ndarray] = None   # [dof]  N·m

    # Task-space commands (ActionSpace.CARTESIAN_POSE / DELTA_EE)
    ee_position:      Optional[jnp.ndarray] = None   # [3]    metres
    ee_orientation:   Optional[jnp.ndarray] = None   # [4]    quaternion (w, x, y, z)
    ee_velocity:      Optional[jnp.ndarray] = None   # [3]    m/s

    # Gripper (0.0 = open, 1.0 = closed)
    gripper:          Optional[float] = None

    # Metadata
    timestamp:        float = 0.0                     # seconds


# ---------------------------------------------------------------------------
# Sensor data
# ---------------------------------------------------------------------------

@dataclass
class SensorData:
    """Raw output from a single ISensor.read() call.

    Backends merge multiple SensorData objects into one Observation.
    Only populate fields that the sensor actually provides.
    """
    rgb:             Optional[jnp.ndarray] = None   # [H, W, 3]  uint8
    depth:           Optional[jnp.ndarray] = None   # [H, W]     float32 metres
    ft_force:        Optional[jnp.ndarray] = None   # [3]  Newtons
    ft_torque:       Optional[jnp.ndarray] = None   # [3]  N·m
    imu_acceleration: Optional[jnp.ndarray] = None  # [3]  m/s²
    imu_angular_velocity: Optional[jnp.ndarray] = None  # [3]  rad/s
    # timestamp_hw:   hardware clock of the sensor (e.g. camera frame timestamp).
    #                 On sim backends, use the sim clock. On real backends, use the
    #                 sensor's own hardware timestamp when available.
    # timestamp_recv: host machine time when this reading was received.
    #                 Backends populate both; pipelines and policies read both.
    timestamp:       float = 0.0
    timestamp_hw:    float = 0.0
    timestamp_recv:  float = 0.0

    # Optional provenance label for timestamp fields (e.g. "sim", "wall", "hardware").
    timestamp_source: str = "unspecified"


# ---------------------------------------------------------------------------
# Task specification types
# ---------------------------------------------------------------------------

@dataclass
class ObsSpec:
    """Declares which observation fields a task requires.

    Backends use this to decide what to compute/render each step.
    Requesting rgb=False on a headless training cluster avoids launching
    the renderer and saves significant GPU memory.
    """
    rgb:          bool = False
    depth:        bool = False
    ft_sensor:    bool = False
    imu:          bool = False
    image_width:  int  = 128
    image_height: int  = 128


@dataclass
class ObjectSpec:
    """A single object to place in the scene."""
    name:        str                            # unique identifier in the scene
    asset_path:  str                            # path to mesh / URDF / SDF
    position:    tuple[float, float, float] = (0.0, 0.0, 0.0)   # metres
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # quaternion


@dataclass
class PropConfig:
    """A scene prop (object) declaration.

    This is the forward-compatible schema used by backends and visualization.
    `ObjectSpec` remains supported for backward compatibility.
    """

    name: str
    asset_path: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass: float = 0.1
    is_fixed: bool = False


@dataclass
class SceneSpec:
    """Full scene description for a task: objects, lighting, table, etc.

    Backends load these assets at initialize() time.
    DomainRandomizer.randomize() modifies poses each episode reset.
    """
    # Preferred representation (matches ARCHITECTURE.md direction).
    props:        list[PropConfig] = field(default_factory=list)
    # Backward-compatible alias used by existing tasks/examples.
    objects:      list[ObjectSpec] = field(default_factory=list)
    table_height: float = 0.0       # metres above world origin
    lighting:     str   = "default" # "default" | "random" | "dark"


@dataclass
class AssetSelection:
    """Records which asset a backend requested/used for a robot."""

    robot_id: str
    requested_format: AssetFormat
    used_format: AssetFormat
    resolved_path: str
    source: Literal["override", "description", "conversion"]
    notes: str = ""


# ---------------------------------------------------------------------------
# Control exceptions
# ---------------------------------------------------------------------------

class HumanInterventionRequired(Exception):
    """Raised by ITask.reset_routine() when a human must intervene.

    RoboEnv and RoboBridge catch this, pause the episode, print the message,
    and wait for the operator to press Enter before continuing.

    Args:
        message: Instructions for the operator (e.g. "Place the red cube
                 at the marked position, then press Enter.").
    """
    pass


class SensorTimeoutError(RuntimeError):
    """Raised by ISensor.read() when a hardware sensor fails to deliver data.

    Caught by SensorBase to return the last valid reading. Propagated only
    if no prior reading has been cached (i.e. the sensor has never succeeded).

    Args:
        sensor_name: Name of the sensor that timed out.
        timeout_s:   The timeout that was exceeded.
    """
    def __init__(self, sensor_name: str, timeout_s: float) -> None:
        super().__init__(
            f"Sensor '{sensor_name}' failed to deliver data within {timeout_s:.3f}s. "
            "Check cable connection and driver status."
        )
        self.sensor_name = sensor_name
        self.timeout_s   = timeout_s


# ---------------------------------------------------------------------------
# Episode metadata
# ---------------------------------------------------------------------------

@dataclass
class EpisodeInfo:
    """Metadata returned by RoboEnv.reset() and RoboEnv.step().

    Contains everything needed to log, debug, or replay an episode.
    """
    episode_id:    int   = 0
    step:          int   = 0
    reward:        float = 0.0
    success:       bool  = False
    failure:       bool  = False
    elapsed_time:  float = 0.0     # wall-clock seconds
    sim_time:      float = 0.0     # simulation seconds
    extra:         dict  = field(default_factory=dict)


MultiTaskMode = Literal["sequential", "concurrent"]


@dataclass
class TaskStepState:
    """Per-task runtime state captured inside EpisodeInfo.extra."""

    task_id: str
    robot_ids: list[str] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    success: bool = False
    failure: bool = False
    step: int = 0
    active: bool = True


@dataclass
class RobotStepState:
    """Per-robot runtime state captured inside EpisodeInfo.extra."""

    robot_id: str
    active_task_id: str = ""
    action: Optional[Action] = None
    obs: Optional[Observation] = None


@dataclass
class ArbitrationEvent:
    """Task-switch event produced by the arbitrator."""

    robot_id: str
    from_task_id: str
    to_task_id: str
    reason: str = ""


@dataclass
class MultiAgentInfo:
    """Structured multi-agent payload stored inside EpisodeInfo.extra."""

    primary_task_id: str = ""
    active_tasks_by_robot: dict[str, str] = field(default_factory=dict)
    task_states: dict[str, TaskStepState] = field(default_factory=dict)
    robot_states: dict[str, RobotStepState] = field(default_factory=dict)
    arbitration_events: list[ArbitrationEvent] = field(default_factory=list)
    rejected_actions: list[dict[str, Any]] = field(default_factory=list)
