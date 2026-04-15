"""Real-hardware backend for the Franka Emika Panda arm via ROS 2 Jazzy.

Activate the ros2_env conda environment before use:
    conda activate ros2_env

Switching between simulation and real is a single-line change:

    # Simulation
    engine = MujocoEngine(robots=["franka"], enable_viewer=True)

    # Real hardware
    engine = FrankaRealBackend(robots=["franka"])

Both implement BaseRobot and return the same Observation/Action types.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:  # JAX not installed in this env (e.g. ros2_env)
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.bridge import BaseRobot
from robodeploy.core.task import BaseTask
from robodeploy.core.types import Action, Observation
from robodeploy.robots.franka.real.ros2_driver import FrankaROS2Driver


class FrankaRealBackend(BaseRobot):
    """Hardware backend for the Franka Panda arm using ROS 2 Jazzy.

    Implements the same BaseRobot interface as MujocoEngine so that
    high-level task/policy code can switch between sim and real with no
    other changes.

    Args:
        robots: Must be ``["franka"]`` (only the Franka Panda is supported).
        config: Optional configuration dict. Recognized keys:
            - ros2_timeout_s (float): Seconds to wait for /joint_states on
              startup (default 10.0).
            - base_frame (str): TF base frame (default "panda_link0").
            - ee_frame (str): TF end-effector frame (default "panda_hand").

    Note:
        Uses NumPy internally for low-latency hardware I/O (per CONTRIBUTING.md),
        then converts to JAX arrays at the Observation boundary for zero-copy
        compatibility with downstream policies.
    """

    _SUPPORTED_ROBOT = "franka"
    _DOF = 7

    def __init__(
        self,
        robots: Optional[list[str]] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(config)

        robots = robots or [self._SUPPORTED_ROBOT]
        if not robots or robots[0] != self._SUPPORTED_ROBOT:
            raise ValueError(
                f"FrankaRealBackend only supports robots=['{self._SUPPORTED_ROBOT}']. "
                f"Got: {robots}"
            )

        self.robots = robots
        self.robot_name = robots[0]
        self._driver = FrankaROS2Driver(config=self.config)

    # ------------------------------------------------------------------
    # BaseRobot interface
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to the robot via ROS 2 and wait for the first joint-state."""
        await super().initialize()
        timeout_s = float(self.config.get("ros2_timeout_s", 10.0))
        self._driver.start(timeout_s=timeout_s)

    async def get_obs(self, tasks: Optional[list[BaseTask]] = None) -> Observation:
        """Read current robot state from ROS 2 and return a unified Observation.

        Executes in NumPy for low latency; converts to JAX arrays before return.
        """
        positions_np, velocities_np, torques_np = self._driver.get_joint_state()
        ee_pos_np, ee_quat_np = self._driver.get_ee_pose()
        gripper_state = self._driver.get_gripper_state()

        # Velocity/angular-velocity from Franka state broadcaster is not always
        # available; zeros are returned until a kinematics module is wired up.
        ee_vel_np = np.zeros(3, dtype=np.float64)
        ee_ang_vel_np = np.zeros(3, dtype=np.float64)

        return Observation(
            joint_positions=jnp.asarray(positions_np, dtype=jnp.float32),
            joint_velocities=jnp.asarray(velocities_np, dtype=jnp.float32),
            joint_torques=jnp.asarray(torques_np, dtype=jnp.float32),
            ee_position=jnp.asarray(ee_pos_np, dtype=jnp.float32),
            ee_orientation=jnp.asarray(ee_quat_np, dtype=jnp.float32),
            ee_velocity=jnp.asarray(ee_vel_np, dtype=jnp.float32),
            ee_angular_velocity=jnp.asarray(ee_ang_vel_np, dtype=jnp.float32),
            gripper_state=gripper_state,
            timestamp=_monotonic_timestamp(),
        )

    async def apply_action(self, action: Action) -> None:
        """Send a joint-position or gripper command to the real robot.

        Only joint-position control is supported. Torque / EE-space commands
        are ignored (add a kinematics/torque module to extend).
        """
        if action.joint_positions is not None:
            positions_np = np.asarray(action.joint_positions, dtype=np.float64)
            self._driver.send_joint_positions(positions_np[: self._DOF])

        if action.gripper is not None:
            self._driver.send_gripper_command(float(action.gripper))

    async def reset(self) -> Observation:
        """Return current observation (no hard reset on real hardware).

        A full reset on real hardware would require a separate home-pose routine.
        Override this method to implement a homing trajectory if needed.
        """
        return await self.get_obs()

    async def shutdown(self) -> None:
        """Cleanly disconnect from ROS 2."""
        self._driver.stop()
        await super().shutdown()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        return {
            "robot": self.robot_name,
            "backend": "real_ros2_jazzy",
            "dof": self._DOF,
            "base_frame": self._driver._base_frame,
            "ee_frame": self._driver._ee_frame,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import time as _time  # noqa: E402  (standard library, intentionally at end)


def _monotonic_timestamp() -> float:
    """Return a monotonic wall-clock timestamp in seconds."""
    return _time.monotonic()
