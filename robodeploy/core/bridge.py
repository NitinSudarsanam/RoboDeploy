"""
Robot bridge module: Base classes for hardware backends.
All backends must inherit from BaseRobot and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .types import Action, Observation


class BaseRobot(ABC):
    """
    Abstract base class for all robot backends (real hardware or simulation).
    
    To add a new hardware backend:
    1. Create a folder in `backends/real/<robot_name>/`
    2. Inherit from this class
    3. Implement `get_obs()` and `apply_action()`
    4. Ensure your `Observation` output matches the fields in `core.types.Observation`
    
    The 100Hz Rule: Control loops must execute in <10ms.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the robot.
        
        Args:
            config: Configuration dictionary (robot-specific parameters)
        """
        self.config = config or {}
        self._is_initialized = False

    async def initialize(self) -> None:
        """
        Initialize the robot hardware/simulation.
        Must be called before any control operations.
        """
        self._is_initialized = True

    async def shutdown(self) -> None:
        """
        Gracefully shut down the robot.
        Clean up hardware resources and connections.
        """
        self._is_initialized = False

    @abstractmethod
    async def get_obs(self) -> Observation:
        """
        Get the current observation from the robot.
        
        Must complete in <10ms to maintain 100Hz control loop.
        All values in SI units (meters, radians, seconds, newtons).
        All arrays must be JAX arrays (jnp.ndarray).
        
        Returns:
            Observation: Current robot state
        """
        raise NotImplementedError

    @abstractmethod
    async def apply_action(self, action: Action) -> None:
        """
        Apply an action to the robot.
        
        Must complete in <10ms to maintain 100Hz control loop.
        
        Args:
            action: The action to apply
        """
        raise NotImplementedError

    async def reset(self) -> Observation:
        """
        Reset the robot to its initial state.
        
        Default implementation calls shutdown and re-initialize.
        Override for custom reset behavior.
        
        Returns:
            Observation: Initial state after reset
        """
        await self.shutdown()
        await self.initialize()
        return await self.get_obs()

    @property
    def is_initialized(self) -> bool:
        """Check if the robot has been initialized."""
        return self._is_initialized
