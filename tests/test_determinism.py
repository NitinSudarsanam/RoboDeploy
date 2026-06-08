from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action
from robodeploy.env import RoboEnv
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask

# Backends excluded from determinism guarantees (documented in GOAL_10 risks):
# - gazebo: server-owned physics, no get/set_sim_state rollback
# - ros2/real: hardware nondeterminism
DETERMINISM_BACKENDS = ("dummy", "mujoco")


def _rollout(env: RoboEnv, *, n_steps: int, seed: int) -> dict:
    try:
        import jax.numpy as jnp
    except Exception:
        import numpy as jnp  # type: ignore[assignment]

    obs, _info = env.reset(seed=seed)
    traj_obs = [np.asarray(obs.joint_positions, dtype=np.float64)]
    actions = []
    rewards = []
    for i in range(n_steps):
        action = Action(joint_positions=jnp.asarray([0.1 * (i + 1), 0.0], dtype=jnp.float32))
        obs, reward, _done, _info = env.step(action)
        traj_obs.append(np.asarray(obs.joint_positions, dtype=np.float64))
        actions.append(np.asarray(action.joint_positions, dtype=np.float64))
        rewards.append(float(reward))
    return {"obs": traj_obs, "actions": actions, "rewards": rewards}


def _make_dummy_env() -> RoboEnv:
    robot = Robot(
        robot_id="robot0",
        description=DummyRobot(),
        tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
    )
    return RoboEnv(backend=DummyBackend(), robots=[robot])


def _assert_rollouts_equal(t1: dict, t2: dict) -> None:
    for a, b in zip(t1["obs"], t2["obs"]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(t1["actions"], t2["actions"]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(t1["rewards"], t2["rewards"], rtol=0.0, atol=0.0)


def _make_mujoco_env() -> RoboEnv | None:
    try:
        import mujoco  # noqa: F401
        import jax.numpy as jnp
    except ImportError:
        return None
    from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
    from robodeploy.core.spaces import ActionSpace, AssetFormat
    from robodeploy.core.types import ObsSpec, SceneSpec
    from robodeploy.description.base import RobotDescription
    from robodeploy.policies.base import PolicyBase
    from robodeploy.tasks.base import TaskBase

    mjcf = """<mujoco model="tiny">
  <worldbody>
    <body name="arm" pos="0 0 0.05">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.01 0.08" fromto="0 0 0 0.12 0 0"/>
      <body name="ee_link" pos="0.12 0 0"><geom type="sphere" size="0.02"/></body>
    </body>
  </worldbody>
  <actuator><position name="joint1" joint="joint1" kp="10"/></actuator>
</mujoco>
"""

    class _TinyDesc(RobotDescription):
        dof = 1
        display_name = "tiny"
        ee_link_name = "ee_link"
        joint_names = ["joint1"]
        joint_position_limits = jnp.asarray([[-3.14, 3.14]], dtype=jnp.float32)
        joint_velocity_limits = jnp.asarray([2.0], dtype=jnp.float32)
        joint_torque_limits = jnp.asarray([10.0], dtype=jnp.float32)
        home_qpos = jnp.asarray([0.0], dtype=jnp.float32)

        def __init__(self, mjcf_path: Path) -> None:
            self._mjcf_path = mjcf_path

        def asset_path(self, fmt, variant: str = "default"):
            del variant
            if fmt == AssetFormat.MJCF:
                return self._mjcf_path
            raise FileNotFoundError(fmt)

    class _TinyTask(TaskBase):
        def obs_spec(self) -> ObsSpec:
            return ObsSpec()

        def scene_spec(self) -> SceneSpec:
            return SceneSpec()

        def language_instruction(self) -> str:
            return ""

        def reset_fn(self, backend) -> None:
            del backend

        def reward_fn(self, obs, action) -> float:
            del obs, action
            return 0.0

        def success_fn(self, obs) -> bool:
            del obs
            return False

    class _Hold(PolicyBase):
        def __init__(self) -> None:
            super().__init__(action_space=ActionSpace.JOINT_POS)

        def get_action(self, obs):
            del obs
            return Action(joint_positions=jnp.asarray([0.0], dtype=jnp.float32))

    td = tempfile.mkdtemp()
    mjcf_path = Path(td) / "tiny.xml"
    mjcf_path.write_text(mjcf, encoding="utf-8")
    robot = Robot(
        robot_id="tiny0",
        description=_TinyDesc(mjcf_path),
        tasks={"t": RobotTask(task=_TinyTask(), policies={"p": _Hold()})},
    )
    return RoboEnv(backend=MuJoCoBackend(config={"enable_viewer": False}), robots=[robot])


def _mujoco_rollout(env: RoboEnv, *, n_steps: int, seed: int) -> dict:
    try:
        import jax.numpy as jnp
    except Exception:
        import numpy as jnp  # type: ignore[assignment]

    obs, _info = env.reset(seed=seed)
    traj_obs = [np.asarray(obs.joint_positions, dtype=np.float64)]
    actions = []
    rewards = []
    for i in range(n_steps):
        action = Action(joint_positions=jnp.asarray([0.05 * (i + 1)], dtype=jnp.float32))
        obs, reward, _done, _info = env.step(action)
        traj_obs.append(np.asarray(obs.joint_positions, dtype=np.float64))
        actions.append(np.asarray(action.joint_positions, dtype=np.float64))
        rewards.append(float(reward))
    return {"obs": traj_obs, "actions": actions, "rewards": rewards}


class DeterminismTests(unittest.TestCase):
    def test_backend_matrix_declares_supported_backends(self):
        self.assertIn("dummy", DETERMINISM_BACKENDS)
        self.assertIn("mujoco", DETERMINISM_BACKENDS)

    def test_two_seeded_rollouts_identical_dummy_backend(self):
        env1 = _make_dummy_env()
        env2 = _make_dummy_env()
        try:
            t1 = _rollout(env1, n_steps=20, seed=42)
            t2 = _rollout(env2, n_steps=20, seed=42)
            _assert_rollouts_equal(t1, t2)
        finally:
            env1.close()
            env2.close()

    def test_two_seeded_rollouts_identical_mujoco_backend(self):
        env1 = _make_mujoco_env()
        if env1 is None:
            self.skipTest("mujoco/jax not installed")
        env2 = _make_mujoco_env()
        assert env2 is not None
        try:
            t1 = _mujoco_rollout(env1, n_steps=30, seed=42)
            t2 = _mujoco_rollout(env2, n_steps=30, seed=42)
            for a, b in zip(t1["obs"], t2["obs"]):
                np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-9)
            for a, b in zip(t1["actions"], t2["actions"]):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(t1["rewards"], t2["rewards"], rtol=0.0, atol=1e-9)
        finally:
            env1.close()
            env2.close()

    def test_different_seeds_produce_different_seed_snapshots(self):
        env1 = _make_dummy_env()
        env2 = _make_dummy_env()
        try:
            env1.reset(seed=1)
            env2.reset(seed=2)
            self.assertEqual(env1.master_seed, 1)
            self.assertEqual(env2.master_seed, 2)
            self.assertNotEqual(env1.seed_snapshot, env2.seed_snapshot)
        finally:
            env1.close()
            env2.close()

    def test_derived_seed_sets_differ(self):
        from robodeploy.core.seeding import derive_seeds

        s1 = derive_seeds(1)
        s2 = derive_seeds(2)
        self.assertNotEqual(s1.env_seed, s2.env_seed)
        self.assertNotEqual(s1.policy_seed, s2.policy_seed)


if __name__ == "__main__":
    unittest.main()
