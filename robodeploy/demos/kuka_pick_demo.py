"""Run a basic Kuka pick demo in MuJoCo MJX."""

from __future__ import annotations

import asyncio

import jax.numpy as jnp

from robodeploy.backends.sim import MujocoEngine
from robodeploy.robots.kuka import ROBOT_NAME
from robodeploy.tasks import BasicKukaPickTask


async def run_demo() -> None:
    engine = MujocoEngine(
        robots=[ROBOT_NAME],
        config={
            "timestep": 0.002,
            "control_hz": 100.0,
            "ee_body": "robot0/ee_link",
            "dof": 7,
            "default_qpos": [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0],
        },
    )
    task = BasicKukaPickTask(robot_id=0)

    await engine.initialize()
    await engine.reset()
    task.reset()

    print("Engine:", engine.get_info())
    print("Instruction:", task.get_instruction())

    for step in range(600):
        obs = await engine.get_obs(tasks=[task])
        action = task.next_action(obs)
        await engine.apply_action(action)

        if step % 100 == 0:
            ee_height = float(obs.ee_position[2])
            q0 = float(obs.joint_positions[0])
            velocity_norm = float(jnp.linalg.norm(obs.joint_velocities))
            print(
                f"step={step:03d} ee_z={ee_height:.3f} q0={q0:.3f} "
                f"|qdot|={velocity_norm:.3f}"
            )

        if task.is_done():
            break

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(run_demo())
