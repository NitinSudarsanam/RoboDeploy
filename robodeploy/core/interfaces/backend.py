"""
IBackend — the hardware/simulator adapter interface.

Every execution environment (MuJoCo, IsaacLab, Genesis, ROS2, LeRobot) is
an IBackend. The backend is responsible for:

  1. Loading the robot model from a RobotDescription.
  2. Loading the scene from a SceneSpec (task assets, table, objects).
  3. Stepping physics or forwarding hardware commands.
  4. Returning a fully-populated Observation each step.

The backend does NOT:
  - Choose control policy.
  - Compute reward or check success (that is the task's job).
  - Normalise observations (that is ObsPipeline's job).
  - Clamp actions for safety (that is SafetyFilter's job).

Adding a new backend:
  1. Create backends/sim/<name>/backend.py or backends/real/<name>/backend.py.
  2. Subclass BackendBase (which subclasses IBackend).
  3. Implement the five abstract methods below.
  4. Decorate with @register_backend("<name>").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types  import Action, Observation, SceneSpec

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.core.interfaces.task   import ITask
    from robodeploy.description.base       import RobotDescription
    from robodeploy.core.robot_config      import RobotConfig


class IBackend(ABC):
    """Abstract adapter between RoboEnv and a physics engine or real robot."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(
        self,
        description: RobotDescription,
        task:        ITask,
        sensors:     list[ISensor],
    ) -> None:
        """Load robot model, scene assets, and sensors.

        Called once before any reset() or step().

        Args:
            description: Static robot definition (joints, limits, asset paths).
            task:        Provides scene_spec() for loading objects into the world.
            sensors:     Sensor instances that need a backend handle to render/read.

        Raises:
            FileNotFoundError: If a required asset cannot be located.
            RuntimeError:      If hardware initialisation fails (real backends).
        """
        ...

    # Multi-robot initializer (optional; default raises if not overridden)
    def initialize_multi(
        self,
        robots: List["RobotConfig"],
        scene:  SceneSpec,
        shared_sensors: List["ISensor"],
    ) -> None:
        """Load multiple robot models, scene assets, and shared sensors."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement initialize_multi()."
        )

    @abstractmethod
    def reset(self) -> Observation:
        """Reset robot and scene to initial state for a new episode.

        Implementations should:
          - Randomise object poses if DomainRandomizer is attached.
          - Return the robot to its home joint configuration.
          - Return a fresh Observation reflecting the reset state.

        Returns:
            Observation: Initial state after reset.
        """
        ...

    # Multi-robot reset (optional; default raises if not overridden)
    def reset_multi(self, robot_ids: List[str] | None = None) -> List[Observation]:
        """Reset all or a subset of robots for a new episode."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement reset_multi()."
        )

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Apply action and advance the simulation or hardware by one control step.

        For sim backends: advances physics by steps_per_control substeps,
        then returns the resulting observation.

        For real backends: publishes the action to hardware controllers,
        waits for the next sensor reading, and returns it as an Observation.

        Args:
            action: Pre-validated, safety-filtered action from the policy.

        Returns:
            Observation: Robot state after the action is applied.
        """
        ...

    # Multi-robot step (optional; default raises if not overridden)
    def step_multi(self, actions: List[Action]) -> List[Observation]:
        """Apply actions and advance all robots by one control step."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement step_multi()."
        )

    @abstractmethod
    def get_obs(self) -> Observation:
        """Return the current observation without advancing physics.

        Used to read state between steps, e.g. for logging or visualisation.

        Returns:
            Observation: Current robot state.
        """
        ...

    # Multi-robot observation (optional; default raises if not overridden)
    def get_obs_multi(self) -> List[Observation]:
        """Return current observations for all robots without advancing physics."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_obs_multi()."
        )

    @abstractmethod
    def close(self) -> None:
        """Release all resources: close viewers, disconnect from hardware, free GPU memory."""
        ...

    # ------------------------------------------------------------------
    # Properties (must be declared by every backend subclass)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def is_real(self) -> bool:
        """True if this backend controls physical hardware, False for simulation.

        Used by RoboEnv to decide whether to engage additional safety checks
        and to disable domain randomization automatically.
        """
        ...

    @property
    @abstractmethod
    def supported_action_spaces(self) -> list[ActionSpace]:
        """Action spaces this backend can accept.

        RoboEnv validates at construction time that the policy's declared
        action_space is in this list.

        Example:
            return [ActionSpace.JOINT_POS, ActionSpace.JOINT_TORQUE]
        """
        ...

    @property
    @abstractmethod
    def control_hz(self) -> float:
        """Control loop frequency in Hz.

        RoboEnv uses this to enforce timing on real backends and to set
        the correct step interval for sim backends.
        """
        ...

    # ------------------------------------------------------------------
    # Optional override: teleportation, physics tuning, multi-robot
    # ------------------------------------------------------------------

    def teleport_object(self, name: str, position: tuple[float, float, float]) -> None:
        """Move a scene object instantly (no physics). Used by DomainRandomizer.

        Default raises NotImplementedError. Override in backends that
        support dynamic scene manipulation.

        Args:
            name:     Object identifier matching ObjectSpec.name.
            position: Target position in metres (x, y, z).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support teleport_object()."
        )

    def set_physics_params(self, **kwargs) -> None:
        """Override physics parameters at runtime (gravity, friction, etc.).

        Used by DomainRandomizer for physics randomisation.
        Default raises NotImplementedError. Override in sim backends.

        Kwargs (backend-specific, documented per backend):
            gravity (list[float]): 3-vector in m/s².
            friction (float):      Global contact friction scale.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_physics_params()."
        )

    def render(self) -> None:
        """Render a viewer frame (no-op if viewer is disabled or backend is real)."""
        pass
