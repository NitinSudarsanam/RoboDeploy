from __future__ import annotations

import tempfile
import unittest

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action, Observation
from robodeploy.demo_recording import DemoRecorder
from robodeploy.env import RoboEnv
from robodeploy.observability.replay import TrajectoryReplayer
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class ReplayDiffTests(unittest.TestCase):
    def test_noiseless_replay_low_divergence(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        recorder = DemoRecorder()
        recorder.metadata["seed"] = 0
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        try:
            obs, _info = env.reset(seed=0)
            for i in range(5):
                action = Action(joint_positions=jnp.asarray([0.1 * (i + 1), 0.0], dtype=jnp.float32))
                obs, reward, done, _info = env.step(action)
                recorder.record_step(obs, action, reward=reward, done=done)
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                path = tmp.name
            recorder.save(path)
            replay_robot = Robot(
                robot_id="robot0",
                description=DummyRobot(),
                tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
            )
            replay_env = RoboEnv(backend=DummyBackend(), robots=[replay_robot])
            try:
                report = TrajectoryReplayer(env=replay_env, recording=path).play()
                self.assertEqual(report.steps_played, 5)
                self.assertLess(report.max_divergence.get("joint_pos", 1.0), 1e-6)
            finally:
                replay_env.close()
        finally:
            env.close()

    def test_checkpoint_file_replay_low_divergence(self):
        from robodeploy.observability.manifest import RunManifest
        from robodeploy.observability.trajectory_checkpoint import TrajectoryCheckpoint

        manifest = RunManifest.for_benchmark_eval(
            benchmark="manipulation_v1/reach_target",
            benchmark_version="1.0",
            policy="scripted",
            backend="dummy",
            seed_base=0,
            n_episodes=1,
        )
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        recorder = DemoRecorder()
        recorder.metadata["seed"] = 5
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        try:
            obs, _info = env.reset(seed=5)
            action = Action(joint_positions=jnp.asarray([0.3, 0.0], dtype=jnp.float32))
            obs, reward, done, _info = env.step(action)
            recorder.record_step(obs, action, reward=reward, done=done)
            checkpoint = TrajectoryCheckpoint.from_episode(
                recorder=recorder,
                manifest=manifest,
                metrics=None,
                episode_index=0,
                seed=5,
            )
            with tempfile.NamedTemporaryFile(suffix=".checkpoint.json", delete=False) as tmp:
                path = tmp.name
            checkpoint.save(path)

            replay_robot = Robot(
                robot_id="robot0",
                description=DummyRobot(),
                tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
            )
            replay_env = RoboEnv(backend=DummyBackend(), robots=[replay_robot])
            try:
                report = TrajectoryReplayer(env=replay_env, recording=path).play()
                self.assertEqual(report.steps_played, 1)
                self.assertLess(report.max_divergence.get("joint_pos", 1.0), 1e-6)
            finally:
                replay_env.close()
        finally:
            env.close()

    def test_divergence_detects_mismatch(self):
        from robodeploy.demo_recording import DemoFrame

        recorder = DemoRecorder()
        recorder.frames = [
            DemoFrame(
                observation={"joint_positions": [9.0, 9.0], "ee_position": [1.0, 0.0, 0.0]},
                action={"joint_positions": [0.1, 0.0]},
                reward=0.0,
                done=False,
            )
        ]
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        try:
            report = TrajectoryReplayer(env=env, recording=recorder, on_divergence="record").play()
            self.assertGreater(report.max_divergence.get("joint_pos", 0.0), 0.01)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
