"""
PolicyBase — shared scaffolding for all policies.

Every concrete policy (RobomimicPolicy, DiffusionPolicy, VLAPolicy,
WaypointPolicy, ...) inherits from PolicyBase rather than IPolicy directly.

PolicyBase provides:
  - A standard __init__ that stores config and the declared action_space.
  - An episode counter incremented by reset().
  - A standard __repr__.

Extension point for batching:
  When VecEnv calls get_action() with a batched Observation [N, ...],
  policies that support it can override get_action_batch() instead.
  PolicyBase.get_action_batch() defaults to a Python loop over get_action()
  so single-robot policies work in a VecEnv without modification.
  High-performance batched policies override get_action_batch() directly.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from robodeploy.core.interfaces.policy import IPolicy
from robodeploy.core.spaces            import ActionSpace
from robodeploy.core.types             import Action, Observation


class PolicyBase(IPolicy):
    """Shared scaffolding for all policies. Subclass this, not IPolicy directly."""

    def __init__(
        self,
        action_space: ActionSpace,
        config:       dict | None = None,
    ) -> None:
        """
        Args:
            action_space: The ActionSpace this policy produces. Stored and
                          returned by the action_space property. Pass the
                          correct space at construction time — RoboEnv
                          validates it against the backend at startup.
            config:       Policy-specific hyperparameters (checkpoint path,
                          smoothing factor, obs key name, etc.).
        """
        self._action_space:   ActionSpace = action_space
        self.config:          dict        = config or {}
        self._episode_count:  int         = 0
        self._instruction:    str         = ""

    # ------------------------------------------------------------------
    # IPolicy implementation
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Increment episode counter and call _reset_impl() for subclass state."""
        self._episode_count += 1
        self._reset_impl()

    @abstractmethod
    def get_action(self, obs: Observation) -> Action:
        """Compute action from observation. Implemented by each policy subclass."""
        ...

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    def set_instruction(self, instruction: str) -> None:
        """Store language instruction. Language-conditioned policies use this."""
        self._instruction = instruction

    @property
    def action_hz(self) -> float:
        """Nominal policy rate used by real-time bridges."""
        return float(self.config.get("action_hz", 0.0))

    def notify_rejected(self, obs: Observation, action: Action) -> None:
        """Hook for sequence policies to stay in sync after rejected actions."""
        return

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _reset_impl(self) -> None:
        """Override to clear internal state (buffers, RNN states, etc.).

        Default is a no-op. Subclasses that maintain episode-scoped state
        must override this.
        """
        pass

    # ------------------------------------------------------------------
    # Batching extension point
    # ------------------------------------------------------------------

    def get_action_batch(self, obs_batch: list[Observation]) -> list[Action]:
        """Compute actions for a batch of observations.

        Default: loop over get_action(). High-performance batched policies
        (e.g. a GPU-vectorised diffusion policy) should override this to
        process the full batch in one forward pass.

        Args:
            obs_batch: List of N observations (one per environment).

        Returns:
            List of N actions.
        """
        return [self.get_action(obs) for obs in obs_batch]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def episode_count(self) -> int:
        """Number of episodes this policy has been reset for."""
        return self._episode_count

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"action_space={self._action_space.name}, "
            f"episodes={self._episode_count})"
        )
