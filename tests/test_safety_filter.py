from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.kinematics.safety import SafetyFilter, SafetyLimits
from robodeploy.safety.violation import SafetyError
from robodeploy.testing import DummyRobot


def _obs(ft_force=None) -> Observation:
    try:
        import jax.numpy as jnp
    except Exception:
        import numpy as jnp  # type: ignore[assignment]
    return Observation(
        joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_angular_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ft_force=None if ft_force is None else jnp.asarray(ft_force, dtype=jnp.float32),
    )


class SafetyFilterTests(unittest.TestCase):
    def test_joint_position_clamp(self):
        filt = DummyRobot().get_safety_filter()
        raw = Action(joint_positions=np.asarray([99.0, -99.0], dtype=np.float32))
        out = filt.filter(raw, ActionSpace.JOINT_POS)
        clamped = np.asarray(out.joint_positions, dtype=np.float64)
        self.assertAlmostEqual(float(clamped[0]), 3.14, places=3)
        self.assertAlmostEqual(float(clamped[1]), -3.14, places=3)
        self.assertTrue(filt.violations())

    def test_workspace_raise_mode(self):
        limits = SafetyLimits(
            joint_position_min=np.array([-3.14, -3.14]),
            joint_position_max=np.array([3.14, 3.14]),
            joint_velocity_max=np.array([2.0, 2.0]),
            workspace_box=(np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])),
        )
        filt = SafetyFilter(limits=limits, on_violation="raise")
        action = Action(ee_position=np.asarray([1.0, 0.0, 0.0], dtype=np.float32), is_delta_ee=True)
        with self.assertRaises(SafetyError):
            filt.filter(action, ActionSpace.DELTA_EE)

    def test_force_halt_engages_estop(self):
        limits = SafetyLimits(
            joint_position_min=np.array([-3.14, -3.14]),
            joint_position_max=np.array([3.14, 3.14]),
            joint_velocity_max=np.array([2.0, 2.0]),
            force_max=10.0,
            control_hz=100.0,
        )
        filt = SafetyFilter(limits=limits, on_violation="halt")
        action = Action(joint_positions=np.asarray([0.0, 0.0], dtype=np.float32))
        filt.filter(action, ActionSpace.JOINT_POS, obs=_obs(ft_force=[100.0, 0.0, 0.0]))
        self.assertTrue(filt.estop_active)

    def test_slew_limits_step_delta(self):
        limits = SafetyLimits(
            joint_position_min=np.array([-3.14, -3.14]),
            joint_position_max=np.array([3.14, 3.14]),
            joint_velocity_max=np.array([1.0, 1.0]),
            control_hz=10.0,
        )
        filt = SafetyFilter(limits=limits)
        first = Action(joint_positions=np.asarray([0.0, 0.0], dtype=np.float32))
        filt.filter(first, ActionSpace.JOINT_POS)
        second = Action(joint_positions=np.asarray([1.0, 0.0], dtype=np.float32))
        out = filt.filter(second, ActionSpace.JOINT_POS)
        delta = float(np.asarray(out.joint_positions, dtype=np.float64)[0])
        self.assertLessEqual(delta, 0.11)


if __name__ == "__main__":
    unittest.main()
