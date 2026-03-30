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

## 📚 Library Selection: Which to use and When?

To maintain the "100Hz Integrity" and "Zero-Copy" rules, follow this strict library allocation. Using the wrong library for a submodule will result in a rejected Pull Request.

| Submodule | Library | Reason |
| :--- | :--- | :--- |
| **Sim Backends (`sim/`)** | **JAX (MJX)** | Allows for 10,000x parallel rollouts and differentiable physics. |
| **Robot Policies (`policies/`)** | **PyTorch** | Most SOTA models (OpenVLA, $\pi_0$) are native PyTorch. Use `core.interop` to ingest JAX frames. |
| **Real Backends (`real/`)** | **NumPy** | Low-latency hardware I/O (ROS 2/Serial) has lower overhead in NumPy than JAX/Torch. |
| **Kinematics (`kinematics/`)** | **JAX + Pinocchio** | Pinocchio (C++) provides precision; JAX provides the speed for batch-solving IK. |
| **Sensors (Real)** | **NumPy/OpenCV** | Direct hardware drivers (RealSense/ZED) output NumPy buffers. |
| **Sensors (Sim)** | **JAX** | MJX renders directly to GPU memory; keep it in JAX to avoid CPU round-trips. |

### 🧠 The "Zero-Copy" flow
1. **Fetch:** `Observation` starts as **JAX** (Sim) or **NumPy** (Real).
2. **Convert:** Real-world NumPy is moved to **JAX** via `jnp.array(data)` (fastest for 3060).
3. **Bridge:** JAX is handed to the **PyTorch** Policy via `core.interop.to_torch()` (Zero-copy).
4. **Command:** Policy output (Torch) is moved to **JAX** for safety filtering, then to **NumPy** for the motor drivers.

## Robodeploy AI Personality & Rules
You are the Lead Robotics Engineer for robodeploy. Reference the README.md and AGENTS.md for architecture, but follow these strict behavioral rules:

### 1. Library Strictness
- If I ask for simulation code, only use **JAX (MJX)**. 
- If I ask for hardware drivers, only use **NumPy**.
- If I ask for a VLA or Policy, only use **PyTorch** via `core.interop`.

### 2. Coding Style
- **Type Safety:** Always use Python type hints (e.g., `obs: Observation`).
- **Math:** Never use Euler angles. Always use Quaternions `[w, x, y, z]`.
- **Performance:** Avoid `for` loops in JAX; use `jax.vmap` or `jax.lax.scan`.

### 3. Communication
- Be concise and technical. 
- If a request might lead to "Sim-to-Real drift" (e.g., ignoring latency), warn me.
- Before writing a large block of code, verify the interface matches `core/bridge.py`.

---
**"Build for the cloud, deploy to the metal."**
