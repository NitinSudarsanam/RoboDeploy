This `CONTRIBUTING.md` is optimized for your **2026 Ubuntu/RTX 3060** setup. It defines the "Rules of the Road" so that any new robot, sensor, or policy added to the library remains compatible with the **Robot-Agnostic Multi-Tasking** architecture.

---

# 🛠️ robodeploy Contribution Guide

Welcome! To hit our **June 2026** deadline, we prioritize **Modularity**, **Zero-Copy Performance**, and **Robot-Agnosticism**. Use this guide to add new features without causing "architecture spiral."

---

## 📏 General Principles
1. **SI Units Only:** Meters, Radians, Seconds, Newtons.
2. **Zero-Copy First:** Never move data to CPU unless logging. Use `core.interop`.
3. **Async Planning:** Robot policies must be non-blocking using `async` generators.
4. **The 100Hz Rule:** Control loops must execute in **<10ms**.

---

## 🤖 Adding a New Robot (`/robots/<name>`)
To add a new hardware platform (e.g., `aloha`, `kuka`), create a new folder in `robots/`.

### 1. Sim Folder (`/sim`)
- **XML naming:** Must contain a self-contained MuJoCo MJCF file (e.g., `panda.xml`).
- **Namespace Safety:** Do **not** hardcode global names for joints/bodies. The `MujocoEngine` will automatically prefix them (e.g., `robot0/`, `robot1/`) during multi-robot composition.
- **Camera Tagging:** You **must** define a camera named `eye_in_hand` or `base_cam` in the XML for the engine to generate batched observations.

### 2. Real Folder (`/real`)
- Implement hardware communication here (e.g., ROS 2 Jazzy nodes).
- **Rule:** The driver must return a `core.types.Observation` object that matches the simulation output format.

### 3. Config File (`config.yaml`)
Define the "Source of Truth" for the robot:
- `dof`: Degrees of Freedom (e.g., 7).
- `default_qpos`: The "home" position array.
- `ee_link`: Name of the end-effector link for Kinematics/IK.

---

## 🧠 Adding a New Robot Policy (`/policies`)
1. Inherit from `core.robot.BaseRobotPolicy`.
2. **Batching Required:** Your `plan()` method must handle a **batch** of observations (e.g., `(N, 224, 224, 3)`).
3. **Multi-Tasking:** Accept a list of `instructions` (strings) of length `N`, where each string corresponds to a specific `robot_id`.

---

## 📚 Library Selection: Which to use and When?

| Submodule | Library | Reason |
| :--- | :--- | :--- |
| **Sim Backend (`backends/sim`)** | **JAX (MJX)** | **Robot-Agnostic.** Must load and simulate any MJCF model from `robots/`. |
| **Real Backend (`backends/real`)** | **NumPy** | **Robot-Specific.** Low-latency hardware I/O (ROS 2/Serial) has lower overhead. |
| **Robot Policies (`policies/`)** | **PyTorch** | Most SOTA models are native PyTorch. Use `core.interop` to ingest JAX frames. |
| **Kinematics (`kinematics/`)** | **JAX + Pinocchio** | Use Pinocchio for math; JAX for batch-solving IK across multiple robots. |

---

## 🧪 Testing & Performance
Before submitting a Pull Request:
- **Physics Sanity:** Run `pytest tests/test_physics.py` to ensure no NaNs or "explosions."
- **3060 Check:** Verify the code runs within the **12GB VRAM** limit of an RTX 3060 using 4-bit quantization for large models.
- **Type Safety:** Run `mypy --strict .` to verify that your robot implementation adheres to the `core.types` contract.

---


You are absolutely right. In 2026, **manually copying and pasting project rules is an "anti-pattern."** You should only have **one** copy of these rules in your repository. GitHub Copilot (and Gemini) are designed to "follow the trail" from your root instructions to your specific guides.

### 1. The "Single Source of Truth" Setup
Instead of pasting the guide everywhere, you create a **pointer** system. This is how you set it up in VS Code so Copilot always knows the rules without you repeating yourself.

**Step A: Create your `CONTRIBUTING.md` once.**
Keep it in your project root. 

**Step B: Create/Update `.github/copilot-instructions.md`**
This is the only file Copilot "forces" itself to read. Instead of the full text, just put this:

```markdown
# Robodeploy AI Rules
You are the Lead Engineer for robodeploy. 

## 🛡️ Mandatory Context
Before generating any code, you MUST reference the following files in the workspace:
1. README.md (For project goals and folder structure)
2. CONTRIBUTING.md (For coding standards, library selection, and robot-agnostic rules)
3. core/types.py (For data structures)

## 🚫 Hard Constraints
- Follow the "Library Selection" table in CONTRIBUTING.md strictly.
- Use JAX for sim, NumPy for real, PyTorch for policies.
- If a request contradicts CONTRIBUTING.md, warn the user before proceeding.
```


## 📝 Adding Heterogeneous Tasks
When a task requires specific sensors (e.g., Depth, Tactile):

1. **Sensor Registration:** Ensure the `robots/<name>/sim/*.xml` has the matching sensor tags.
2. **The Observation Contract:** The `MujocoEngine` will always return a "Full Suite" observation PyTree. 
   - If your task doesn't use `depth`, the engine will provide a zero-tensor for that index.
   - Your Policy must check `obs.task_mask` to know which sensors are valid for which robot.
3. **Interacting with Sim:** Tasks can use `sim_engine.set_property(name, value)` to spawn new objects or change colors mid-simulation for dynamic testing.


## Direct Interaction (The "Simulator API")
To interact with the simulator during a task (e.g., clicking to move an object or changing gravity), you add an Interaction Hook in the MujocoEngine.

```Python
# Move an object manually to test robot robustness
sim_engine.teleport_object("red_cube", position=[0.5, 0.2, 0.0])

# Change physics properties for domain randomization
sim_engine.set_physics(gravity=[0, 0, -1.0]) # "Low gravity" task
```

---



## 💡 Pro-Tip: Using AI with this Guide
When using **Gemini Pro** or **Copilot** in VS Code:
1. Always keep this file open in a tab.
2. Use the `@workspace` command to ask: *"Based on CONTRIBUTING.md, implement the UR5 real-world driver in robots/ur5/real/."*

### 2. How to "Summon" the Guide in Chat
Now, when you are coding, you don't need to copy anything. Use these three "Power Shortcuts":

* **The Implicit Way:** Because of the `.github/copilot-instructions.md` file, Copilot is *already* reading the contributing guide in the background. You just talk to it normally.
* **The Explicit Way (The `#` Symbol):** If you want to be 100% sure it's following the rules for a specific task:
    > "Create a new robot driver for a Kuka arm. Follow **#CONTRIBUTING.md** and **#README.md**."
* **The `@workspace` Way:**
    > "**@workspace** /explain how I should add a new depth-based task based on our contributing guidelines."



---
**"Build for the cloud, deploy to the metal."**