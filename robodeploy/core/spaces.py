"""
Action and observation space enumerations.

ActionSpace defines the control mode a policy produces and a backend consumes.
Every backend declares which spaces it supports. Every policy declares which
space its outputs live in. RoboEnv validates compatibility at construction time
so mismatches surface immediately — not at step 10,000 of a training run.
"""

from enum import Enum, auto


class ActionSpace(Enum):
    """Control modes a policy may produce.

    JOINT_POS      — absolute joint position targets [rad]
    JOINT_VEL      — joint velocity targets [rad/s]
    JOINT_TORQUE   — direct joint torque commands [N·m]
    CARTESIAN_POSE — absolute end-effector pose (position + quaternion)
    DELTA_EE       — relative end-effector displacement (position delta + axis-angle)
    """
    JOINT_POS      = auto()
    JOINT_VEL      = auto()
    JOINT_TORQUE   = auto()
    CARTESIAN_POSE = auto()
    DELTA_EE       = auto()


class AssetFormat(Enum):
    """Robot description asset formats supported by backends.

    URDF  — canonical source; supported by ROS2, Pinocchio, MuJoCo (via conversion)
    MJCF  — MuJoCo XML; hand-tuned physics, used by MuJoCoBackend
    USD   — Universal Scene Description; used by IsaacLab / IsaacGym
    """
    URDF = "urdf"
    MJCF = "mjcf"
    USD  = "usd"
