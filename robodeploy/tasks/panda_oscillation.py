"""Sinusoidal oscillation task for the Franka Panda arm.

Replicates the joint-space oscillation from the standalone move_panda.py script
inside the RoboDeploy task API so it can run on any BaseRobot backend (sim or real).

Uses only NumPy — no JAX dependency — so it is safe to import in ros2_env.

Joint limits (from panda.xml):
    joint1: [-2.9,  2.9]   joint5: [-2.9,  2.9]
    joint2: [-1.8,  1.8]   joint6: [-0.1,  3.7]
    joint3: [-2.9,  2.9]   joint7: [-2.9,  2.9]
    joint4: [-3.0,  0.0]
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from robodeploy.core.task import BaseTask
from robodeploy.core.types import Action, Observation

# Per-joint: (ctrl_min, ctrl_max) matching panda.xml actuator ctrlranges.
_PANDA_ARM_CTRL_RANGES: list[tuple[float, float]] = [
    (-2.9,  2.9),   # joint 1
    (-1.8,  1.8),   # joint 2
    (-2.9,  2.9),   # joint 3
    (-3.0,  0.0),   # joint 4
    (-2.9,  2.9),   # joint 5
    (-0.1,  3.7),   # joint 6
    (-2.9,  2.9),   # joint 7
]

# Frequency and phase parameters (matching move_panda.py)
_OSCILLATION_FREQ_HZ: float = 0.8 / (2.0 * math.pi)  # ≈ 0.127 Hz per joint
_PHASE_INCREMENT: float = 0.7          # radians between adjacent joints
_AMPLITUDE_FRACTION: float = 0.25     # fraction of total range


class PandaOscillationTask(BaseTask):
    """Drives the Panda arm through periodic sinusoidal joint-space motion.

    Each joint oscillates between ``center ± amplitude`` where:
        center    = 0.5 * (ctrl_min + ctrl_max)
        amplitude = 0.25 * (ctrl_max - ctrl_min)

    The gripper opens and closes at half the arm oscillation frequency.

    This task never terminates (``is_done()`` always returns False); use it
    inside a timed loop or add a step-count limit via the ``max_steps`` arg.

    Args:
        robot_id:   Which robot index this task controls (default 0).
        timestep_s: Physics timestep in seconds, used to convert steps → time.
                    Must match the engine's ``timestep`` config value.
        max_steps:  Optional step limit; ``is_done()`` returns True after this
                    many calls to ``next_action()``.  ``None`` = run forever.
    """

    def __init__(
        self,
        robot_id: int = 0,
        timestep_s: float = 0.002,
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__(robot_id=robot_id)
        self._timestep_s = timestep_s
        self._max_steps = max_steps
        self._step = 0

        # Pre-compute center and amplitude for each arm joint (plain numpy)
        self._centers = np.array(
            [0.5 * (lo + hi) for lo, hi in _PANDA_ARM_CTRL_RANGES],
            dtype=np.float32,
        )
        self._amplitudes = np.array(
            [_AMPLITUDE_FRACTION * (hi - lo) for lo, hi in _PANDA_ARM_CTRL_RANGES],
            dtype=np.float32,
        )
        self._phases = np.array(
            [_PHASE_INCREMENT * i for i in range(len(_PANDA_ARM_CTRL_RANGES))],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def get_observation_spec(self) -> dict:
        return {"rgb": False, "depth": False, "segmentation": False}

    def get_instruction(self) -> str:
        return "Oscillate all Panda arm joints sinusoidally."

    def reset(self) -> None:
        self._step = 0

    def next_action(self, obs: Observation) -> Action:
        """Compute the next sinusoidal joint target given the current step."""
        del obs  # not used — motion is purely time-based

        t = self._step * self._timestep_s
        self._step += 1

        # joint_i target = center_i + amp_i * sin(2π * freq * t + phase_i)
        omega = 2.0 * math.pi * _OSCILLATION_FREQ_HZ
        angle = float(omega * t)
        joint_positions = self._centers + self._amplitudes * np.sin(
            np.float32(angle) + self._phases
        )

        # Gripper: 0.5 + 0.5 * sin(0.5 * t) → [0, 1]  (0=open, 1=closed)
        gripper = float(0.5 + 0.5 * math.sin(0.5 * t))

        return Action(joint_positions=joint_positions, gripper=gripper)

    def is_done(self) -> bool:
        if self._max_steps is None:
            return False
        return self._step >= self._max_steps
