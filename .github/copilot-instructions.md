# RoboDeploy AI guidance

Read the local code before changing it. The current public surface is centered on `RoboEnv`, `Robot`, `RobotTask`, `Action`, `Observation`, backend factories, robot descriptions under `robodeploy/description/*`, and the component registries.

Keep simulator and hardware details behind backends. Robot descriptions should provide assets and metadata; they should not open MuJoCo, ROS2, serial buses, or Isaac Sim directly.

Use explicit conversions at array boundaries. The project uses NumPy broadly, optional JAX arrays in shared dataclasses, and optional PyTorch in policy code. Do not assume strict library separation or zero-copy DLPack unless the implementation actually provides it.

When adding optional integrations, import optional dependencies lazily and raise actionable `ImportError` messages. Registered placeholders should be named honestly, usually with a `_stub` suffix.