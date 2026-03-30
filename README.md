# 🤖 robodeploy: The Unified Robot Bridge

**robodeploy** is a high-performance, research-first library designed for the next generation of Embodied AI. It provides a seamless, zero-copy bridge between JAX-accelerated simulations (MuJoCo MJX) and real-world Franka Emika Panda hardware.

## 🌟 Key Features
- **Unified Robot API:** Support for VLAs, Diffusion Policies, and World Models via a single robot-centric interface.
- **MJX Native:** Leverage MuJoCo XLA for 10,000x parallel simulation on a single GPU.
- **Zero-Copy Interop:** DLPack-based memory sharing between JAX (Sim) and PyTorch (Inference).
- **Hardware Agnostic:** Switch from `sim` to `real` via a single configuration toggle.
- **Safety Guaranteed:** Integrated Pinocchio-based kinematic filters for real-world collision avoidance.

## 📂 Project Structure
```text
robodeploy/
├── core/               # The "Contract": Types, Base Classes, & Interop
│   ├── types.py        # Shared Observation/Action dataclasses
│   ├── bridge.py       # Abstract Base Classes for Hardware Backends
│   ├── robot.py        # Universal Robot Policy/Brain interface
│   └── interop.py      # JAX <-> Torch zero-copy utilities
├── backends/           # The "Muscles": Execution layers
│   ├── sim/            # MuJoCo MJX implementations
│   └── real/           # ROS 2 Jazzy hardware drivers
├── sensors/            # The "Eyes": Modular perception
│   ├── camera/         # MJX-rendered vs. RealSense/ZED
│   └── tactile/        # Force/Torque abstractions
├── policies/           # The "Brains": Robot Policy implementations
│   ├── vla/            # Reactive models (OpenVLA, Pi-0)
│   ├── world_models/   # Predictive models (Cosmos, Genie)
│   └── diffusion/      # Generative trajectories
├── kinematics/         # The "Math": Pinocchio IK & Safety filters
└── tests/              # CI/CD: Physics sanity & benchmark suite
```

## 🚀 Quick Start
Install the environment in 60 seconds using `uv`:
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
uv sync --extra sim --extra real
```


## 🛠️ Usage Example
```python
from robodeploy.backends.sim import MujocoFranka
from robodeploy.policies.vla import OpenVLARobotPolicy

# Initialize the hardware backend
franka = MujocoFranka(mode="sim") # Flip to "real" for hardware

# Initialize the robot brain (policy)
brain = OpenVLARobotPolicy(model_id="openvla-7b")

obs = franka.get_obs()
async for action in brain.plan(obs):
    franka.apply_action(action)
```
