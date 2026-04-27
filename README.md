# 🤖 robodeploy: The Unified Robot Bridge

**robodeploy** is a high-performance, research-first library designed for the next generation of Embodied AI. It provides a seamless, zero-copy bridge between JAX-accelerated simulations (MuJoCo MJX) and real-world hardware backends.

## 🌟 Key Features
- **Robot-Agnostic Sim:** A universal MuJoCo MJX engine. Load any robot by name from the `robots/` library.
- **Multi-Robot Batching:** Native support for controlling fleets of heterogeneous robots in a single JAX world.
- **Unified Robot Policy API:** Support for VLAs, Diffusion Policies, and World Models via a single interface.
- **Zero-Copy Interop:** DLPack-based memory sharing between JAX (Sim/Perception) and PyTorch (Inference).
- **Hardware Agnostic:** Seamlessly toggle between `sim` and `real` backends using the same high-level code.

## 📂 Project Structure
```text
robodeploy/
├── core/               # The "Contract": Types, Base Classes, & Interop
│   ├── types.py        # Shared Observation/Action/Task dataclasses
│   ├── bridge.py       # Base classes for Robot Backends (Sim/Real)
│   ├── robot.py        # Universal Robot Policy (Brain) interface
│   └── interop.py      # JAX <-> Torch zero-copy utilities
├── robots/             # 🛠️ ADD NEW ROBOTS HERE
│   ├── franka/         
│   │   ├── sim/        # panda.xml, meshes, and MJX configs
│   │   ├── real/       # ROS 2 Jazzy hardware drivers
│   │   └── config.yaml # Joint limits, home positions, and metadata
│   └── ur5/            
│       ├── sim/        
│       └── real/
├── backends/           # The "Muscles": Execution layers
│   ├── sim/            # MujocoEngine (The Robot-Agnostic Physics Runner)
│   └── real/           # ROS 2 Jazzy hardware glue
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
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra sim --extra real
```

## Backend setup (MuJoCo, ROS 2 + RViz, Isaac Sim)

See [docs/BACKEND_SETUP.md](docs/BACKEND_SETUP.md) and the curated example index in [examples/README.md](examples/README.md).

## 🛠️ Usage Example: Multi-Robot, Multi-Task
```python
from robodeploy.backends.sim import MujocoEngine
from robodeploy.tasks import DepthAlignmentTask, RGBSortingTask

# 1. Spawn robots
sim_engine = MujocoEngine(robots=["franka", "ur5"])

# 2. Assign heterogeneous tasks
tasks = [
    DepthAlignmentTask(robot_id=0, instruction="Align to the recessed hole"),
    RGBSortingTask(robot_id=1, instruction="Put red blocks in the blue bin")
]

# 3. The Engine uses the Task list to gather observations
# It returns a BatchedObservation where fields are padded or masked
obs = sim_engine.get_obs(tasks=tasks)

# 4. Your Policy (VLA) processes the heterogeneous batch
actions = brain.plan(obs) 

sim_engine.apply_action(actions)
```

