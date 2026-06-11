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

from robodeploy.core.spaces import ActionSpace, AssetFormat

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
    images:               dict[str, jnp.ndarray] = field(default_factory=dict)
    depths:               dict[str, jnp.ndarray] = field(default_factory=dict)

    # --- Force / torque sensor (populated when hardware present) -----------
    ft_force:             Optional[jnp.ndarray] = None   # [3]  Newtons
    ft_torque:            Optional[jnp.ndarray] = None   # [3]  N·m
    ft_forces:            dict[str, jnp.ndarray] = field(default_factory=dict)
    ft_torques:           dict[str, jnp.ndarray] = field(default_factory=dict)

    # --- IMU (populated when hardware present) ------------------------------
    imu_acceleration:     Optional[jnp.ndarray] = None   # [3]  m/s²
    imu_angular_velocity: Optional[jnp.ndarray] = None   # [3]  rad/s

    # --- Gripper (0.0 = fully open, 1.0 = fully closed) --------------------
    gripper_state:        Optional[float] = None

    # --- Perception (populated by sensors / perception transforms) ---------
    # Sensor FK path (ee_pose rig); policies should prefer over ee_position when set.
    ee_pose:              Optional[jnp.ndarray] = None   # [3]    metres
    ee_pose_orientation:  Optional[jnp.ndarray] = None   # [4]    wxyz
    # objects[name] = (position_xyz, orientation_wxyz)
    objects:              dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = field(
        default_factory=dict
    )
    # contact_state[sensor_name] = binary touch (ContactSensor or FT threshold)
    contact_state:        dict[str, bool] = field(default_factory=dict)
    sensor_status:        dict[str, str] = field(default_factory=dict)
    metadata:             dict[str, Any] = field(default_factory=dict)
    camera_frames:        dict[str, str] = field(default_factory=dict)
    camera_intrinsics:    dict[str, dict[str, float]] = field(default_factory=dict)
    camera_extrinsics:    dict[str, dict[str, object]] = field(default_factory=dict)
    # metadata: fused scores / transform outputs (e.g. grasp_stability)
    metadata:             dict[str, Any] = field(default_factory=dict)

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
    language_instruction: Optional[str] = None


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
    action_space:     Optional[ActionSpace] = None
    is_delta_ee:      bool = False


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
    ft_forces:       dict[str, jnp.ndarray] = field(default_factory=dict)
    ft_torques:      dict[str, jnp.ndarray] = field(default_factory=dict)
    imu_acceleration: Optional[jnp.ndarray] = None  # [3]  m/s²
    imu_angular_velocity: Optional[jnp.ndarray] = None  # [3]  rad/s
    objects: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = field(
        default_factory=dict
    )
    contact_state: dict[str, bool] = field(default_factory=dict)
    ee_pose: Optional[jnp.ndarray] = None
    ee_pose_orientation: Optional[jnp.ndarray] = None
    status: str = "ok"
    frame_id: Optional[str] = None
    intrinsics: Optional[dict[str, float]] = None
    extrinsics: Optional[dict[str, object]] = None
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

@dataclass(frozen=True)
class Pose3D:
    """Rigid transform in world or parent frame (metres + unit quaternion wxyz)."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class RobotInit:
    """Normalized multi-robot initialization payload for backends."""

    robot_id: str
    description: Any
    base_pose: Pose3D = field(default_factory=Pose3D)
    namespace: str | None = None
    sensor_rig: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class SensorMount:
    """Pose of a sensor relative to a robot link or the world frame."""

    parent_link: Optional[str] = None
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CameraRequest:
    """Named camera fields requested by a task or policy."""

    name: str
    width: int = 640
    height: int = 480
    fields: tuple[Literal["rgb", "depth"], ...] = ("rgb",)


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
    objects:      bool = False
    image_width:  int  = 128
    image_height: int  = 128
    cameras:      list[CameraRequest] = field(default_factory=list)


def validate_observation(
    obs: Observation,
    spec: ObsSpec,
    *,
    policy: str = "warn",
) -> None:
    """Check that required ObsSpec fields are present on an observation."""
    missing: list[str] = []
    if spec.rgb and obs.rgb is None and not obs.images:
        missing.append("rgb")
    if spec.depth and obs.depth is None and not obs.depths:
        missing.append("depth")
    if spec.ft_sensor and obs.ft_force is None and not obs.ft_forces:
        missing.append("ft_force")
    if spec.imu and obs.imu_acceleration is None:
        missing.append("imu_acceleration")
    if spec.objects and not obs.objects:
        missing.append("objects")
    if not missing:
        return
    msg = f"Observation missing required ObsSpec fields: {', '.join(missing)}"
    mode = str(policy).lower()
    if mode == "raise":
        raise ValueError(msg)
    if mode == "warn":
        import warnings

        warnings.warn(msg, RuntimeWarning, stacklevel=3)


@dataclass
class ObjectSpec:
    """A single object to place in the scene."""
    name:        str                            # unique identifier in the scene
    asset_path:  str                            # path to mesh / URDF / SDF
    position:    tuple[float, float, float] = (0.0, 0.0, 0.0)   # metres
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # quaternion


@dataclass
class GeomSpec:
    """Procedural or mesh-backed geometry declaration for a scene prop."""

    kind: Literal["box", "cylinder", "sphere", "capsule", "mesh", "plane"]
    size: tuple[float, ...]
    mesh_path: Optional[str] = None


@dataclass
class MaterialSpec:
    """Visual and contact material parameters shared across backends."""

    rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0)
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    texture: Optional[str] = None


@dataclass
class PropConfig:
    """A scene prop (object) declaration.

    This is the forward-compatible schema used by backends and visualization.
    `ObjectSpec` remains supported for backward compatibility.
    """

    name: str
    asset_path: str = ""
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass: float = 0.1
    is_fixed: bool = False
    geom: Optional[GeomSpec] = None
    material: MaterialSpec = field(default_factory=MaterialSpec)
    asset: Optional[dict[AssetFormat, str]] = None
    parent_link: Optional[str] = None
    inertia_diag: Optional[tuple[float, float, float]] = None
    collision_layer: int = 0
    collision_mask: int = 0xFFFF
    friction_dist: Optional[tuple[float, float]] = None


@dataclass
class LightSpec:
    """Scene light declaration."""

    position: tuple[float, float, float] = (0.0, 0.0, 2.0)
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    diffuse: tuple[float, float, float] = (0.8, 0.8, 0.8)
    kind: Literal["directional", "point", "spot"] = "directional"


@dataclass
class CameraSpec:
    """World or link-mounted camera declaration."""

    name: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    fov_deg: float = 60.0
    resolution: tuple[int, int] = (640, 480)
    parent_link: Optional[str] = None


@dataclass
class TerrainSpec:
    """Terrain declaration for simulation backends."""

    kind: Literal["flat", "heightfield", "procedural"] = "flat"
    size: tuple[float, float] = (4.0, 4.0)
    heightfield_path: Optional[str] = None
    procedural_params: Optional[dict] = None


@dataclass
class WorldSpec:
    """Backend-loadable world description."""

    props: list[PropConfig] = field(default_factory=list)
    lights: list[LightSpec] = field(default_factory=list)
    cameras: list[CameraSpec] = field(default_factory=list)
    terrain: TerrainSpec = field(default_factory=TerrainSpec)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)


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
    world:        WorldSpec = field(default_factory=WorldSpec)

    def to_world(self) -> WorldSpec:
        """Return a normalized world view, preserving legacy props/objects."""

        props: list[PropConfig] = []
        seen: set[str] = set()
        for prop in [*self.world.props, *self.props]:
            if prop.name in seen:
                continue
            props.append(prop)
            seen.add(prop.name)
        for obj in self.objects:
            if obj.name in seen:
                continue
            props.append(
                PropConfig(
                    name=obj.name,
                    asset_path=obj.asset_path,
                    position=obj.position,
                    orientation=obj.orientation,
                )
            )
            seen.add(obj.name)

        return WorldSpec(
            props=props,
            lights=list(self.world.lights),
            cameras=list(self.world.cameras),
            terrain=self.world.terrain,
            gravity=self.world.gravity,
        )

    def to_ir(self):
        """Return backend-agnostic Scene IR for cross-backend validation."""
        from robodeploy.core.scene_ir import world_to_ir

        preset = self.lighting if self.lighting in ("minimal", "bright", "dark", "randomized") else None
        return world_to_ir(self.to_world(), lighting_preset=preset)


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
