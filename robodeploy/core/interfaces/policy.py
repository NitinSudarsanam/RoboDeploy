"""
IPolicy — the control policy (brain) interface.

A policy takes an Observation and returns an Action. It has no knowledge of
which backend is running or whether the robot is simulated or real. This is
the entire point: a policy trained in MuJoCo should be deployable to a
physical robot by swapping only the backend.

Policy responsibilities:
  - Maintain any internal state (history buffer, hidden RNN state, etc.).
  - Produce an Action whose populated fields match its declared action_space.
  - Be fast: get_action() must complete within the backend's control period.

Policy non-responsibilities:
  - Safety clamping (SafetyFilter handles this).
  - Observation normalisation (ObsPipeline handles this).
  - Episode lifecycle (RoboEnv handles this).

Adding a new policy:
  1. Create policies/learned/<name>.py or policies/scripted/<name>.py.
  2. Subclass PolicyBase (which subclasses IPolicy).
  3. Implement reset() and get_action().
  4. Decorate with @register_policy("<name>").
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types  import Action, Observation


class IPolicy(ABC):
    """Abstract policy: maps Observation → Action."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> None:
        """Reset internal policy state at the start of a new episode.

        Must clear:
          - Observation history buffers.
          - RNN / transformer hidden states.
          - Action smoothing buffers.
          - Any episode-scoped counters.

        Called by RoboEnv.reset() automatically before the first step.
        """
        ...

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @abstractmethod
    def get_action(self, obs: Observation) -> Action:
        """Compute the next action given the current observation.

        Must complete within the backend's control period (1 / control_hz).
        For 100 Hz control that is 10 ms. If inference is slower, the caller
        (RoboBridge for real hardware) will hold the last action.

        Args:
            obs: Current robot observation, already normalised by ObsPipeline.

        Returns:
            Action with fields populated for the declared action_space.
            Do NOT clamp joint limits here — SafetyFilter handles that.
        """
        ...

    # ------------------------------------------------------------------
    # Properties (declared by subclasses)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        """The ActionSpace this policy outputs.

        RoboEnv validates at construction time that the backend supports
        this space. Example: return ActionSpace.JOINT_POS
        """
        ...

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def set_instruction(self, instruction: str) -> None:
        """Provide a language instruction for language-conditioned policies (VLAs).

        Default is a no-op. Override in VLAPolicy, DiffusionPolicy, etc.

        Args:
            instruction: Natural language task description, e.g.
                         "Pick the red cube and place it in the bin."
        """
        pass

    def warmup(self, obs: Observation) -> None:
        """Run a dummy forward pass to trigger JIT compilation before deployment.

        Default calls get_action(obs) and discards the result.
        Override if the policy needs a different warmup sequence.

        Args:
            obs: A representative observation (e.g. from env.reset()).
        """
        self.get_action(obs)

    def get_action_batch(self, obs_batch: list[Observation]) -> list[Action]:
        """Optional batch inference hook for multi-robot task configs."""
        return [self.get_action(obs) for obs in obs_batch]

    @property
    def action_hz(self) -> float:
        """Nominal action production rate used by real-time bridges."""
        return 0.0

    def notify_rejected(self, obs: Observation, action: Action) -> None:
        """Called when an action is rejected before reaching the backend."""
        return
