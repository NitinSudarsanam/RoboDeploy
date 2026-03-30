
# 🛠️ robodeploy Contribution Guide

Welcome! To hit our **June 2026** deadline, we prioritize **Modularity**, **Zero-Copy Performance**, and **Robot-Agnosticism**. Use this guide to add new features without causing "architecture spiral."

---

## 🏗️ Architectural Mandate: Strict Independence

### 1. The "Interchangeable Backend" Rule
Switching from Sim to Real must be a simple change in arguments at the top-level script. 
- **Decoupling:** A robot definition in `robots/` must never know if it is in a simulator or on hardware. 
- **No Circular Imports:** The `robots/` folder must never import from `backends/`. The `backends/` handle the robots, not the other way around.

### 2. Dependency Injection
Never hardcode a simulator or hardware driver inside a robot class. 
- **Good:** `engine = MujocoEngine(robots=["franka"])`
- **Bad:** `class Franka: def __init__(self): self.sim = Mujoco()`

---

## ✍️ Coding Standards & Hygiene

### 1. File Granularity
- **One Task, One File:** If a file exceeds **300 lines**, it must be split into logical sub-modules.
- **Logic vs. Data:** Keep MJCF/XML strings in `.xml` files, configurations in `.yaml`, and only logic in `.py`.
- **Meaningful Naming:** - Use `joint_velocities` instead of `qv`, `end_effector_pose` instead of `ee`.
    - Booleans must be prefixed: `is_calibrated`, `has_gripper`, `should_reset`.
    - Suffix non-SI units: `timeout_ms`, `frequency_hz`.

### 2. Function Design
- **Single Responsibility:** A function should do one thing. If you need "and" to describe it, split it.
- **Type Hinting:** All functions **must** include type hints for all parameters and return values.

---

## 🤖 Adding a New Robot (`/robots/<name>`)

### 1. Sim Folder (`/sim`)
- **Namespace Safety:** Do **not** hardcode global names for joints/bodies. The `MujocoEngine` prefixes them (e.g., `robot0/`) during composition.
- **Camera Tagging:** You **must** define a camera named `eye_in_hand` or `base_cam` for batched observations.

### 2. Real Folder (`/real`)
- Implement hardware communication here (e.g., ROS 2 Jazzy nodes).
- **Rule:** Drivers must return a `core.types.Observation` matching the simulation format exactly.

---

## 📝 Adding Heterogeneous Tasks
1. **Sensor Registration:** Ensure the `robots/<name>/sim/*.xml` has matching sensor tags.
2. **The Observation Contract:** The `MujocoEngine` always returns a "Full Suite" observation PyTree. 
   - If a task doesn't use `depth`, the engine provides a zero-tensor for that index.
   - Policies must check `obs.task_mask` to know which sensors are valid.
3. **Simulator API (Direct Interaction):** Use hooks in `MujocoEngine` for manual manipulation:
```python
sim_engine.teleport_object("red_cube", position=[0.5, 0.2, 0.0])
sim_engine.set_physics(gravity=[0, 0, -9.81])
```

---

## 📚 Library Selection (Strict)

| Submodule | Library | Reason |
| :--- | :--- | :--- |
| **Sim Backend** | **JAX (MJX)** | **Robot-Agnostic.** Must be `@jit` compatible. |
| **Real Backend** | **NumPy** | **Robot-Specific.** Low-latency hardware I/O. |
| **Robot Policies** | **PyTorch** | AI Inference. Use `core.interop` for zero-copy. |
| **Kinematics** | **JAX + Pinocchio** | Use Pinocchio for math; JAX for batch-solving. |

---

## 🧪 Testing & Performance
- **Physics Sanity:** Run `pytest tests/test_physics.py` to ensure no NaNs.
- **3060 Check:** Verify code runs within **12GB VRAM** using 4-bit quantization.
- **100Hz Rule:** Control loops must execute in **<10ms**.

---

## 💡 Pro-Tip: Using AI with this Guide
When using **Gemini Pro** or **Copilot** in VS Code:
1. Use **#CONTRIBUTING.md** and **#README.md** to provide context.
2. Use **@workspace** to ensure the AI follows the "Strict Independence" rule when creating new files.

**"Build for the cloud, deploy to the metal."**