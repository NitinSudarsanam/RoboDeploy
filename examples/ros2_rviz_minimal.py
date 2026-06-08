"""Minimal ROS2Backend + RViz wiring example (structure test).

This example is meant to be run in a ROS2 Python environment (rclpy available)
with the appropriate robot graph running (e.g., franka_ros2 or ros2_control).

It demonstrates:
- per-robot namespaces (/robot0, /robot1)
- RViz marker publishing from SceneSpec + task viz_goals()
- N:M:K wiring shape (2 robots, 2 tasks, shared backend)
"""

from __future__ import annotations

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from robodeploy.backends.simulator import backend_for_simulator  # noqa: E402
from robodeploy.core.robot import Robot, RobotTask  # noqa: E402
from robodeploy.core.spaces import ActionSpace  # noqa: E402
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec  # noqa: E402
from robodeploy.description.franka.description import FrankaDescription  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402
from robodeploy.policies.base import PolicyBase  # noqa: E402
from robodeploy.tasks.base import TaskBase  # noqa: E402


class DummyHoldPolicy(PolicyBase):
    def __init__(self, home) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS, config={"action_hz": 50.0})
        self._home = home

    def _reset_impl(self) -> None:
        return

    def get_action(self, obs: Observation) -> Action:
        del obs
        return Action(joint_positions=self._home)


class DummyVizTask(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec()

    def scene_spec(self) -> SceneSpec:
        # Minimal: empty scene; RViz will still show EE pose.
        return SceneSpec()

    def language_instruction(self) -> str:
        return "Hold position."

    def reset_fn(self, backend) -> None:
        return

    def reward_fn(self, obs, action) -> float:
        return 0.0

    def success_fn(self, obs) -> bool:
        return False

    def viz_goals(self, obs=None):
        # Example goal marker: EE pose target at origin
        return [{"kind": "pose", "position": [0.5, 0.0, 0.5], "orientation": [1, 0, 0, 0], "label": "target"}]


def main() -> None:
    d0 = FrankaDescription()
    d1 = FrankaDescription()

    r0 = Robot(
        robot_id="robot0",
        description=d0,
        tasks={
            "t0": RobotTask(
                task=DummyVizTask(),
                policies={"hold": DummyHoldPolicy(d0.home_qpos)},
                mode="sequential",
            )
        },
    )
    r1 = Robot(
        robot_id="robot1",
        description=d1,
        tasks={
            "t1": RobotTask(
                task=DummyVizTask(),
                policies={"hold": DummyHoldPolicy(d1.home_qpos)},
                mode="sequential",
            )
        },
    )

    backend = backend_for_simulator(
        "ros2_rviz",
        robots=[r0, r1],
        config_overrides={
            "robot0.joint_pos_cmd_topic": "joint_group_impedance_controller/commands",
            "robot1.joint_pos_cmd_topic": "joint_group_impedance_controller/commands",
        },
    )

    env = RoboEnv(backend=backend, robots=[r0, r1])
    env.reset()
    for _ in range(50):
        env.step()
    env.close()


if __name__ == "__main__":
    main()

