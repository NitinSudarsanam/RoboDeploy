"""
ITask — the task/scene definition interface.

A task defines:
  1. What the world looks like (SceneSpec: objects, poses).
  2. What the robot is trying to do (language instruction).
  3. Which observations are needed (ObsSpec).
  4. How success and failure are evaluated (reward, success, failure functions).
  5. How the scene is reset between episodes (reset_fn).

Tasks are backend-agnostic. They describe the problem, not the physics.
The backend is responsible for loading and instantiating the SceneSpec.

Tasks are the right place for:
  - Domain randomisation configuration (pose ranges, object variants).
  - Reward shaping logic.
  - Success criteria (e.g. object lifted above threshold).
  - Language instructions for language-conditioned policies.

Tasks are NOT the right place for:
  - Physics parameter setting (backend handles that).
  - Camera rendering (sensors handle that).
  - Policy inference (policy handles that).

Adding a new task:
  1. Create tasks/manipulation/<name>.py (or tasks/navigation/<name>.py, etc.).
  2. Subclass TaskBase (which subclasses ITask).
  3. Implement all abstract methods.
  4. Decorate with @register_task("<name>").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec

if TYPE_CHECKING:
    from robodeploy.core.interfaces.backend import IBackend


class ITask(ABC):
    """Abstract task: defines the problem the robot must solve."""

    # ------------------------------------------------------------------
    # Static declarations (called once at construction time)
    # ------------------------------------------------------------------

    @abstractmethod
    def obs_spec(self) -> ObsSpec:
        """Declare which observation fields this task requires.

        The backend only computes/renders what is declared here.
        Returning rgb=False on a headless server saves significant GPU memory.

        Returns:
            ObsSpec describing required sensors and image dimensions.

        Example:
            return ObsSpec(rgb=True, depth=False, image_width=224, image_height=224)
        """
        ...

    @abstractmethod
    def scene_spec(self) -> SceneSpec:
        """Declare the scene: which objects exist and their initial poses.

        Called once at backend.initialize(). The backend loads all assets
        listed here. DomainRandomizer.randomize() can vary poses each reset.

        Returns:
            SceneSpec with the full list of objects and scene configuration.
        """
        ...

    @abstractmethod
    def language_instruction(self) -> str:
        """Return the natural language goal for this task.

        Used by language-conditioned policies (VLAs). Should be a complete
        sentence describing the desired outcome.

        Returns:
            str: e.g. "Pick the red cube and place it in the blue bin."
        """
        ...

    # ------------------------------------------------------------------
    # Episode lifecycle (called every episode)
    # ------------------------------------------------------------------

    @abstractmethod
    def reset_fn(self, backend: IBackend) -> None:
        """Randomise and reset the scene for a new episode.

        Called by RoboEnv after backend.reset(). Use backend.teleport_object()
        to reposition objects. Use DomainRandomizer here for varied object
        poses, lighting, and physics.

        Args:
            backend: Active backend, used to manipulate the scene.
        """
        ...

    # ------------------------------------------------------------------
    # Evaluation (called every step)
    # ------------------------------------------------------------------

    @abstractmethod
    def reward_fn(self, obs: Observation, action: Action) -> float:
        """Compute the scalar reward for the current step.

        Should be fast (called every control step). Avoid heavy computation.

        Args:
            obs:    Observation after the action was applied.
            action: Action that was applied this step.

        Returns:
            float: Scalar reward. Sign convention: higher is better.
        """
        ...

    @abstractmethod
    def success_fn(self, obs: Observation) -> bool:
        """Return True if the task has been successfully completed.

        Args:
            obs: Current observation.

        Returns:
            bool: True triggers episode termination with success=True.
        """
        ...

    @abstractmethod
    def failure_fn(self, obs: Observation) -> bool:
        """Return True if the episode should terminate due to failure.

        Examples: robot fell, joint limit violated, object dropped off table.

        Args:
            obs: Current observation.

        Returns:
            bool: True triggers episode termination with success=False.
        """
        ...

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def reset_routine(self, backend: "IBackend") -> Iterator[Action]:
        """Yield a safe trajectory to reset the scene for a new episode.

        This replaces fire-and-forget reset_fn for real hardware where
        joints cannot be teleported.

        In SIMULATION: default implementation calls reset_fn() and returns
        immediately (empty iterator). The backend teleports everything.

        On REAL HARDWARE: override to yield a sequence of joint-position
        Actions that move the robot back to home, then raise
        HumanInterventionRequired if a human must reset physical props
        (e.g. reposition an object that was dropped).

        RoboEnv and RoboBridge both call this method. RoboEnv executes
        yielded actions via backend.step(). RoboBridge feeds them into
        the ActionBuffer so the ControlLoop handles them at full frequency.

        Args:
            backend: Active backend. Use backend.is_real to branch if needed.

        Yields:
            Action: One action per step toward the home / reset configuration.

        Raises:
            HumanInterventionRequired: Signal that a human must intervene
                before the episode can start. RoboEnv will pause and prompt.

        Example (real hardware override):
            def reset_routine(self, backend):
                # Move arm to home
                yield from self._plan_to_home(backend.get_obs())
                # Prompt researcher to replace objects
                raise HumanInterventionRequired(
                    "Place the red cube at the marked position, then press Enter."
                )
        """
        # Default: teleport (sim) — call reset_fn and yield nothing
        self.reset_fn(backend)
        return
        yield   # make this a generator even when it returns early

    def max_steps(self) -> int:
        """Maximum number of steps per episode before forced termination.

        Default is 1000. Override to set task-specific episode length.
        """
        return 1000

    # ------------------------------------------------------------------
    # Optional hooks used by RoboEnv / RoboBridge
    # ------------------------------------------------------------------

    def _on_reset(self) -> None:
        """Hook called by RoboEnv at episode start.

        TaskBase overrides this to manage an internal step counter. Custom task
        implementations that do not inherit from TaskBase can ignore this.
        """
        return

    def _on_step(self) -> None:
        """Hook called by RoboEnv/RoboBridge after each step."""
        return

    @property
    def step_count(self) -> int:
        """Number of steps elapsed in the current episode (default 0)."""
        return 0

    def on_activate(self) -> None:
        """Called when this task becomes active for a robot."""
        return

    def on_deactivate(self) -> None:
        """Called when this task yields control for a robot."""
        return
