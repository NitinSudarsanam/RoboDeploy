from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action
from robodeploy.env import RoboEnv
from robodeploy.tasks.randomization import (
    DomainRandomizerConfig,
    RandomLevel,
    SensorNoiseConfig,
    build_dr_config_from_cell,
    dr_config_to_dict,
    resolve_domain_randomizer_config,
    resolve_random_level,
)
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask
from robodeploy.training.dr_sweep import DRSweep, DRSweepConfig, iter_dr_sweep_cells


class DRSweepConfigResolutionTests(unittest.TestCase):
    def test_resolve_random_level_from_string(self):
        self.assertEqual(resolve_random_level("full"), RandomLevel.FULL)

    def test_resolve_domain_randomizer_config_parses_nested_dict(self):
        cfg = resolve_domain_randomizer_config(
            {
                "level": "FULL",
                "physics": {"friction_range": (0.6, 1.4)},
                "sensor_noise": {"joint_pos_std": 0.002},
            }
        )
        assert cfg is not None
        self.assertEqual(cfg.level, RandomLevel.FULL)
        self.assertEqual(cfg.physics.friction_range, (0.6, 1.4))
        self.assertEqual(cfg.sensor_noise.joint_pos_std, 0.002)

    def test_build_dr_config_from_cell_scales_noise(self):
        base = DomainRandomizerConfig(sensor_noise=SensorNoiseConfig(joint_pos_std=0.01))
        cell = {"level": "FULL", "sensor_noise_scale": 2.0, "position_range": 0.03}
        out = build_dr_config_from_cell(cell, base=base)
        self.assertEqual(out.sensor_noise.joint_pos_std, 0.02)

    def test_dr_config_round_trip_dict(self):
        cfg = DomainRandomizerConfig(level=RandomLevel.LIGHT, seed=7)
        restored = resolve_domain_randomizer_config(dr_config_to_dict(cfg))
        assert restored is not None
        self.assertEqual(restored.level, RandomLevel.LIGHT)
        self.assertEqual(restored.seed, 7)

    def test_iter_dr_sweep_cells_cartesian_product(self):
        config = DRSweepConfig(
            levels=[RandomLevel.NONE, RandomLevel.FULL],
            object_position_ranges=[(0.0, 0.0), (0.05, 0.05)],
            physics_friction_ranges=[(1.0, 1.0)],
            sensor_noise_scales=[0.0, 1.0],
        )
        cells = iter_dr_sweep_cells(config)
        self.assertEqual(len(cells), 2 * 2 * 1 * 2)


class DRSweepRunTests(unittest.TestCase):
    def _make_env(self, dr_cfg: DomainRandomizerConfig, seed: int):
        task = DummyTask(
            config={
                "domain_randomization": {
                    "level": dr_cfg.level.name,
                    "seed": seed,
                }
            }
        )
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=task, policies={"p": DummyPolicy(0.0)})},
        )
        return RoboEnv(backend=DummyBackend(), robots=[robot])

    def test_dr_sweep_produces_report(self):
        sweep = DRSweep(
            env_fn=self._make_env,
            policy_fn=lambda _env: lambda obs: Action(
                joint_positions=np.asarray([0.1, 0.1], dtype=np.float32)
            ),
            config=DRSweepConfig(
                n_seeds=1,
                n_episodes_per_seed=1,
                max_steps_per_episode=3,
                levels=[RandomLevel.NONE, RandomLevel.LIGHT],
                object_position_ranges=[(0.0, 0.0)],
                physics_friction_ranges=[(1.0, 1.0)],
                sensor_noise_scales=[0.0],
            ),
        )
        report = sweep.run()
        self.assertEqual(len(report.cells), 2)
        self.assertIn("success_rate", report.cells[0])
        self.assertIn("level", report.sensitivity or report.cells[0]["params"])


if __name__ == "__main__":
    unittest.main()
