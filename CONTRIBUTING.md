# 🛠️ robodeploy Contribution Guide

Welcome! To hit our June 2026 deadline, we prioritize **Modularity** and **Interoperability**. Use this guide to add new features.

## 📏 General Principles
1. **SI Units Only:** Meters, Radians, Seconds, Newtons.
2. **Zero-Copy First:** Never move data to CPU unless logging. Use `core.interop`.
3. **Async Planning:** Robot policies must be non-blocking using `async` generators.
4. **The 100Hz Rule:** Control loops must execute in <10ms.

## 🤖 Adding a New Hardware Backend
1. Create a folder in `backends/real/<robot_name>/`.
2. Inherit from `core.bridge.BaseRobot`.
3. Implement `get_obs()` and `apply_action()`.
4. Ensure your `Observation` output matches the fields in `core.types.Observation`.

## 🧠 Adding a New Robot Policy (Brain)
1. Inherit from `core.robot.BaseRobotPolicy`.
2. Implement the `async def plan(self, obs)` generator.
3. If using PyTorch, use `core.interop.to_torch(obs.rgb)` to ingest JAX frames without latency.

## 📷 Adding a New Sensor
1. Inherit from `core.sensor.BaseCamera`.
2. Provide a `sim/` implementation (MuJoCo renderer) and a `real/` implementation (Hardware driver).
3. All camera outputs must be returned as `jnp.ndarray` (JAX Arrays).

## 🧪 Testing Requirements
Before submitting a PR:
- Run `pytest tests/test_physics.py` to ensure no "physics explosions" (NaNs).
- Run `mypy --strict .` to verify type safety.
- Verify your code runs on an NVIDIA 3060 (12GB VRAM) or higher.

---
**"Build for the cloud, deploy to the metal."**
