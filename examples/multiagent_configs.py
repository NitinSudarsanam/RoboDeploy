"""Structure-only usage examples for N:M:K RoboEnv/RoboBridge configs."""

from __future__ import annotations

from robodeploy import RoboEnv
from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action
from robodeploy.description.franka import FrankaDescription
from robodeploy.description.kuka import KukaDescription
from robodeploy.policies.learned.diffusion import DiffusionPolicy
from robodeploy.policies.learned.robomimic import RobomimicPolicy
from examples.tasks.peg_insertion import PegTask
from examples.tasks.pick_place import PickPlaceTask
from examples.tasks.pour import PourTask


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
            Robot(
                robot_id="franka",
                description=FrankaDescription(),
                tasks={
                    "pick": RobotTask(
                        task=PickPlaceTask(),
                        policies={"robomimic": RobomimicPolicy(checkpoint_path="pick_place.pt")},
                        mode="sequential",
                    )
                },
            ),
            Robot(
                robot_id="kuka",
                description=KukaDescription(),
                tasks={
                    "pour": RobotTask(
                        task=PourTask(),
                        policies={"diffusion": DiffusionPolicy()},
                        mode="sequential",
                    )
                },
            ),
        ],
    )


def many_robots_one_task_shared_policy(shared_policy) -> RoboEnv:
    # NOTE: Sharing one policy instance across robots is usually not desired if
    # the policy maintains internal state; this is structure-only.
    return RoboEnv(
        backend=MuJoCoBackend(),
        robots=[
            Robot(
                robot_id="franka",
                description=FrankaDescription(),
                tasks={
                    "coop_pick": RobotTask(
                        task=PickPlaceTask(),
                        policies={"shared": shared_policy},
                        mode="sequential",
                    )
                },
            ),
            Robot(
                robot_id="kuka",
                description=KukaDescription(),
                tasks={
                    "coop_pick": RobotTask(
                        task=PickPlaceTask(),
                        policies={"shared": shared_policy},
                        mode="sequential",
                    )
                },
            ),
        ],
    )


def one_robot_many_tasks_sequential() -> RoboEnv:
    env = RoboEnv(
        backend=MuJoCoBackend(),
        robots=[
            Robot(
                robot_id="franka",
                description=FrankaDescription(),
                tasks={
                    "pick": RobotTask(
                        task=PickPlaceTask(),
                        policies={"robomimic": RobomimicPolicy(checkpoint_path="pick.pt")},
                        mode="sequential",
                    ),
                    "peg": RobotTask(
                        task=PegTask(),
                        policies={"diffusion": DiffusionPolicy()},
                        mode="sequential",
                    ),
                },
                task_weights={"pick": 1.0, "peg": 0.0},
            )
        ],
    )
    env.switch_task("franka", "peg", reason="operator_selected")
    return env


def one_robot_many_tasks_concurrent() -> RoboEnv:
    # Robot requires at least one sequential task; we keep "pick" sequential and
    # run "pour" concurrently as the background behavior.
    return RoboEnv(
        backend=MuJoCoBackend(),
        robots=[
            Robot(
                robot_id="franka",
                description=FrankaDescription(),
                tasks={
                    "pick": RobotTask(
                        task=PickPlaceTask(),
                        policies={"p": DiffusionPolicy()},
                        mode="sequential",
                    ),
                    "pour": RobotTask(
                        task=PourTask(),
                        policies={"p": DiffusionPolicy()},
                        mode="concurrent",
                    ),
                },
                task_weights={"pick": 1.0},
                task_action_resolver=average_joint_position_actions,
            )
        ],
    )

