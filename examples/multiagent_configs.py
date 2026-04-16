"""Structure-only usage examples for N:M:K RoboEnv/RoboBridge configs."""

from __future__ import annotations

from robodeploy import RoboEnv
from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
from robodeploy.core.robot_config import RobotConfig
from robodeploy.core.task_config import TaskConfig
from robodeploy.core.types import Action
from robodeploy.description.franka import FrankaDescription
from robodeploy.description.kuka import KukaDescription
from robodeploy.policies.learned.diffusion import DiffusionPolicy
from robodeploy.policies.learned.robomimic import RobomimicPolicy
from robodeploy.tasks.manipulation.peg_insertion import PegTask
from robodeploy.tasks.manipulation.pick_place import PickPlaceTask
from robodeploy.tasks.manipulation.pour import PourTask


def average_joint_position_actions(robot_id: str, actions: list[Action]) -> Action:
    del robot_id
    valid = [action.joint_positions for action in actions if action.joint_positions is not None]
    if not valid:
        return Action()
    merged = sum(valid) / len(valid)
    return Action(joint_positions=merged)


def many_robots_many_tasks_many_policies() -> RoboEnv:
    return RoboEnv(
        backend=MuJoCoBackend(),
        robots=[
            RobotConfig(description=FrankaDescription(), robot_id="franka"),
            RobotConfig(description=KukaDescription(), robot_id="kuka"),
        ],
        tasks=[
            TaskConfig(
                task=PickPlaceTask(),
                robot_ids=["franka"],
                policy=RobomimicPolicy(checkpoint_path="pick_place.pt"),
                task_id="pick",
                mode="sequential",
            ),
            TaskConfig(
                task=PourTask(),
                robot_ids=["kuka"],
                policy=DiffusionPolicy(),
                task_id="pour",
                mode="sequential",
            ),
        ],
    )


def many_robots_one_task_shared_policy(shared_policy) -> RoboEnv:
    return RoboEnv(
        backend=MuJoCoBackend(),
        robots=[
            RobotConfig(description=FrankaDescription(), robot_id="franka"),
            RobotConfig(description=KukaDescription(), robot_id="kuka"),
        ],
        tasks=[
            TaskConfig(
                task=PickPlaceTask(),
                robot_ids=["franka", "kuka"],
                policy=shared_policy,
                task_id="coop_pick",
                mode="concurrent",
            ),
        ],
    )


def one_robot_many_tasks_sequential() -> RoboEnv:
    env = RoboEnv(
        backend=MuJoCoBackend(),
        robots=[RobotConfig(description=FrankaDescription(), robot_id="franka")],
        tasks=[
            TaskConfig(
                task=PickPlaceTask(),
                robot_ids=["franka"],
                policy=RobomimicPolicy(checkpoint_path="pick.pt"),
                task_id="pick",
                mode="sequential",
            ),
            TaskConfig(
                task=PegTask(),
                robot_ids=["franka"],
                policy=DiffusionPolicy(),
                task_id="peg",
                mode="sequential",
            ),
        ],
    )
    env.switch_task("franka", "peg", reason="operator_selected")
    return env


def one_robot_many_tasks_concurrent() -> RoboEnv:
    return RoboEnv(
        backend=MuJoCoBackend(),
        robots=[RobotConfig(description=FrankaDescription(), robot_id="franka")],
        tasks=[
            TaskConfig(
                task=PickPlaceTask(),
                robot_ids=["franka"],
                policy=DiffusionPolicy(),
                task_id="pick",
                mode="concurrent",
                action_resolver="average_joint_pos",
            ),
            TaskConfig(
                task=PourTask(),
                robot_ids=["franka"],
                policy=DiffusionPolicy(),
                task_id="pour",
                mode="concurrent",
                action_resolver="average_joint_pos",
            ),
        ],
        action_resolvers={
            "average_joint_pos": average_joint_position_actions,
        },
    )

