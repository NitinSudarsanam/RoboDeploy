from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action
from robodeploy.demo_recording import DemoRecorder
from robodeploy.env import RoboEnv
from robodeploy.observability.manifest import RunManifest
from robodeploy.observability.trajectory_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    TrajectoryCheckpoint,
    write_trajectory_checkpoint,
)
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class TrajectoryCheckpointTests(unittest.TestCase):
    def test_round_trip_and_schema_version(self):
        manifest = RunManifest.for_benchmark_eval(
            benchmark="manipulation_v1/reach_target",
            benchmark_version="1.0",
            policy="scripted",
            backend="dummy",
            seed_base=0,
            n_episodes=1,
        )
        recorder = DemoRecorder()
        recorder.metadata["seed"] = 7
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        try:
            obs, _info = env.reset(seed=7)
            action = Action(joint_positions=jnp.asarray([0.1, 0.0], dtype=jnp.float32))
            obs, reward, done, _info = env.step(action)
            recorder.record_step(obs, action, reward=reward, done=done)
        finally:
            env.close()

        checkpoint = TrajectoryCheckpoint.from_episode(
            recorder=recorder,
            manifest=manifest,
            metrics=None,
            episode_index=0,
            seed=7,
            episode_id="ep-0",
        )
        self.assertEqual(checkpoint.schema_version, CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(len(checkpoint.frames), 1)
        self.assertEqual(checkpoint.manifest["benchmark"], "manipulation_v1/reach_target")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / checkpoint.default_filename()
            checkpoint.save(path)
            loaded = TrajectoryCheckpoint.load(path)
            self.assertEqual(loaded.seed, 7)
            self.assertEqual(loaded.episode_id, "ep-0")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("manifest", payload)
            self.assertIn("frames", payload)

    def test_checkpoint_replay_and_manifest_round_trip(self):
        manifest = RunManifest.for_benchmark_eval(
            benchmark="manipulation_v1/reach_target",
            benchmark_version="1.0",
            policy="scripted",
            backend="dummy",
            seed_base=0,
            n_episodes=1,
        )
        recorder = DemoRecorder()
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        try:
            obs, _info = env.reset(seed=11)
            action = Action(joint_positions=jnp.asarray([0.2, 0.0], dtype=jnp.float32))
            obs, reward, done, _info = env.step(action)
            recorder.record_step(obs, action, reward=reward, done=done)
        finally:
            env.close()

        checkpoint = TrajectoryCheckpoint.from_episode(
            recorder=recorder,
            manifest=manifest,
            metrics=None,
            episode_index=0,
            seed=11,
        )
        restored_manifest = checkpoint.run_manifest()
        self.assertEqual(restored_manifest.benchmark, "manipulation_v1/reach_target")
        self.assertEqual(restored_manifest.package_version, manifest.package_version)

        replay_robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        replay_env = RoboEnv(backend=DummyBackend(), robots=[replay_robot])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / checkpoint.default_filename()
                checkpoint.save(path)
                from robodeploy.observability.replay import TrajectoryReplayer

                report = TrajectoryReplayer(env=replay_env, recording=path).play()
                self.assertEqual(report.steps_played, 1)
                self.assertLess(report.max_divergence.get("joint_pos", 1.0), 1e-6)
        finally:
            replay_env.close()

    def test_write_trajectory_checkpoint_helper(self):
        manifest = RunManifest.for_benchmark_eval(
            benchmark="manipulation_v1/reach_target",
            benchmark_version="1.0",
            policy="scripted",
            backend="dummy",
            seed_base=1,
            n_episodes=2,
        )
        recorder = DemoRecorder()
        with tempfile.TemporaryDirectory() as tmp:
            out = write_trajectory_checkpoint(
                out_dir=tmp,
                recorder=recorder,
                manifest=manifest,
                metrics=None,
                episode_index=1,
                seed=42,
            )
            self.assertTrue(out.is_file())
            self.assertIn("episode_0001_seed42", out.name)


if __name__ == "__main__":
    unittest.main()
