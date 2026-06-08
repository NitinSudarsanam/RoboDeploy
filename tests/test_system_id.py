from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robodeploy.calibration.system_id.friction import FrictionEstimator, FrictionSample
from robodeploy.calibration.system_id.mass import PayloadMassEstimator
from robodeploy.calibration.system_id.pipeline import SystemIdPipeline
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.env import RoboEnv
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class SystemIdTests(unittest.TestCase):
    def test_friction_fit(self):
        estimator = FrictionEstimator()
        samples = [
            FrictionSample(velocity_rad_s=0.1, torque_Nm=0.5),
            FrictionSample(velocity_rad_s=0.2, torque_Nm=0.7),
            FrictionSample(velocity_rad_s=-0.1, torque_Nm=-0.5),
        ]
        params = estimator.fit(samples)
        self.assertGreater(abs(params.coulomb_Nm), 0.0)

    def test_pipeline_dummy_env(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        env.reset()
        with tempfile.TemporaryDirectory() as tmp:
            from robodeploy.calibration.store import CalibrationStore

            pipeline = SystemIdPipeline(store=CalibrationStore(root=Path(tmp)))
            result = pipeline.run(env, joint_indices=[0], robot_id="test_robot")
            self.assertIn("joint_0", result.friction or {})
            loaded = pipeline.store.load("system_id", robot_id="test_robot")
            self.assertIn("payload_mass_kg", loaded)
        env.close()

    def test_mass_estimator_returns_non_negative(self):
        estimator = PayloadMassEstimator(arm_length_m=0.3)
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        env.reset()
        mass = estimator.estimate(env)
        self.assertGreaterEqual(mass, 0.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
