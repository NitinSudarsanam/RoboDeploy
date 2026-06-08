"""Example DR sweep workflow — evaluate policy robustness across randomization axes.

Usage (from repo root)::

    python -m examples.sim2real.run_dr_sweep --preset kuka_pick_mujoco --output reports/dr_sweep_001/

With dummy backend (no MuJoCo)::

    python -m examples.sim2real.run_dr_sweep --dummy --output reports/dr_sweep_dummy/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robodeploy.sim2real.config import merge_preset_with_dr
from robodeploy.tasks.randomization import RandomLevel
from robodeploy.training.dr_sweep import DRSweep, DRSweepConfig


def _make_env_factory(*, preset: str, dummy: bool):
    from robodeploy.tasks.randomization import DomainRandomizerConfig

    if dummy:
        from robodeploy.core.robot import Robot, RobotTask
        from robodeploy.env import RoboEnv
        from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask

        def env_fn(dr_cfg: DomainRandomizerConfig, seed: int):
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

        return env_fn

    from robodeploy.builtins import import_builtins
    from examples.config import load_example_preset
    from robodeploy.env import RoboEnv

    import_builtins()
    base_preset = load_example_preset(preset)

    def env_fn(dr_cfg: DomainRandomizerConfig, seed: int):
        cfg = merge_preset_with_dr(base_preset, dr_cfg)
        task_kwargs = dict(cfg.get("task_kwargs") or {})
        task_kwargs.setdefault("domain_randomization", {})["seed"] = seed
        cfg["task_kwargs"] = task_kwargs
        return RoboEnv.from_config(cfg)

    return env_fn


def _policy_factory(env):
    del env
    return lambda obs: None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="examples.sim2real.run_dr_sweep")
    parser.add_argument("--preset", default="kuka_pick_mujoco", help="Sim training preset name.")
    parser.add_argument("--dummy", action="store_true", help="Use DummyBackend (no MuJoCo).")
    parser.add_argument("--output", required=True, help="Output directory for sweep JSON.")
    parser.add_argument("--seeds", type=int, default=2, help="Seeds per sweep cell.")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per seed.")
    parser.add_argument("--steps", type=int, default=20, help="Max steps per episode.")
    args = parser.parse_args(argv)

    sweep = DRSweep(
        env_fn=_make_env_factory(preset=str(args.preset), dummy=bool(args.dummy)),
        policy_fn=_policy_factory,
        config=DRSweepConfig(
            n_seeds=int(args.seeds),
            n_episodes_per_seed=int(args.episodes),
            max_steps_per_episode=int(args.steps),
            levels=[RandomLevel.NONE, RandomLevel.LIGHT, RandomLevel.FULL],
            object_position_ranges=[(0.0, 0.0), (0.02, 0.02)],
            physics_friction_ranges=[(1.0, 1.0), (0.8, 1.2)],
            sensor_noise_scales=[0.0, 1.0],
        ),
    )
    report = sweep.run()
    out = Path(args.output)
    report.write_json(out / "dr_sweep_report.json")
    print(report.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
