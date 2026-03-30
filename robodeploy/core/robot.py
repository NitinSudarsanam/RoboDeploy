"""
Robot policy module: Base classes for robot control policies (brains).
All policies must inherit from BaseRobotPolicy and implement the plan() method.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from .types import Action, Observation


class BaseRobotPolicy(ABC):
    """
    Abstract base class for all robot policies/brains.
    
    To add a new robot policy:
    1. Inherit from this class
    2. Implement the `async def plan(self, obs)` generator
    3. If using PyTorch, use `core.interop.to_torch(obs.rgb)` to ingest JAX frames without latency
    
    Policies must be non-blocking using async generators.
    Must maintain 100Hz control loop: each step <10ms.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the policy.
        
        Args:
            config: Configuration dictionary (policy-specific hyperparameters)
        """
        self.config = config or {}

    @abstractmethod
    async def plan(self, obs: Observation) -> AsyncGenerator[Action, None]:
        """
        Plan actions based on observations.
        
        Must be an async generator that yields actions.
        Can maintain state across multiple steps.
        
        Args:
            obs: Current robot observation
            
        Yields:
            Action: Planned action for the robot
            
        Example:
            async def plan(self, obs):
                while True:
                    # Process observation
                    action = self._compute_action(obs)
                    yield action
                    # Receive next observation
                    obs = yield  # (implicit in async generators)
        """
        raise NotImplementedError

    async def reset(self) -> None:
        """
        Reset the policy state (e.g., clear buffers, reset hidden states).
        
        Override if your policy maintains internal state.
        """
        pass
