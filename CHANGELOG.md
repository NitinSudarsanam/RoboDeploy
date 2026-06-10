# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-08

### Added

- MuJoCo multi-robot support via `MuJoCoBackend.initialize_multi()` (namespaced MJCF, shared scene).
- `Pose3D`, `RobotInit`, and optional `Robot.base_pose` for multi-robot placement.
- `robodeploy.multirobot.resolvers` with `average_joint_actions`, `weighted_blend`, and registry helpers.
- Multi-robot YAML preset `two_franka_pick_mujoco` and runnable example under `examples/multirobot/`.
- `RoboEnv.from_config()` multi-robot dict specs in `cfg["robots"]`.
- Entry-point plugin discovery (`robodeploy.discover()` / `auto_discover_entry_points()`).
- `robodeploy assets verify` CLI for shipped asset SHA256 checks.
- PyPI publish workflow (tag `v*` upload + `workflow_dispatch` dry-run), Docker CPU image sketch, conda recipe template, and `docs/PLUGINS.md`.
- `docs/RELEASE.md` with local `python -m build` dry-run and first-release checklist.
- Example third-party plugin package at `examples/plugin_robot_demo/`.
- CI: `test_plugin_discovery`, `two_franka_pick_mujoco` E2E, and `assets verify` CLI tests.

### Changed

- Version bumped from `0.1.0` to `0.2.0`.
- CI unittest matrix extended across Python 3.10–3.12 and Linux / Windows / macOS.

## [0.1.0] - 2025-12-01

### Added

- Initial RoboDeploy runtime: `RoboEnv`, registry, MuJoCo / ROS2 backends, example presets.

[0.2.0]: https://github.com/anthropic-ai/robodeploy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/anthropic-ai/robodeploy/releases/tag/v0.1.0
