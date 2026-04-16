"""
ITransform — composable observation transforms.

Replaces the `is_real` flag that was previously branched inside ObsPipeline.
Instead of one pipeline that internally checks whether it's running in sim or
real, you construct *two different pipelines* from the same transform library:

    # Sim pipeline (training): add synthetic noise, then normalize
    sim_pipeline = ObsPipeline([
        GaussianNoiseTransform(joint_std=0.001, ee_std=0.001),
        NormalizeTransform.from_dataset(dataset),
    ])

    # Real pipeline (deployment): only normalize — real hardware already has noise
    real_pipeline = ObsPipeline([
        NormalizeTransform.from_dataset(dataset),   # same statistics as sim
    ])

Both pipelines live in user code. ObsPipeline itself has no `is_real` flag.
The configuration is the only difference — the execution path is identical.

This pattern also makes unit testing trivial: test each transform in isolation
with a synthetic Observation rather than mocking a full backend.

Built-in transforms provided here cover the common cases. Implement ITransform
for anything custom — a single method to override.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Observation


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class ITransform(ABC):
    """A single, stateless or stateful transform applied to an Observation.

    Transforms are composable. ObsPipeline holds an ordered list of them and
    calls forward() in sequence. Implementing a custom transform requires
    overriding only forward().

    fit() is optional — only stateful transforms (like NormalizeTransform)
    need to compute statistics from data. Stateless transforms (like noise
    injection) ignore it.
    """

    @abstractmethod
    def forward(self, obs: Observation) -> Observation:
        """Apply this transform to obs and return a new Observation.

        Must not mutate obs in place — return a new object.
        Must be fast: called every control step on the hot path.

        Args:
            obs: Input observation.

        Returns:
            Transformed observation.
        """
        ...

    def fit(self, dataset: list[Observation]) -> None:
        """Compute statistics from a dataset of observations (optional).

        Override for stateful transforms that need to compute mean/std from data
        before they can run. No-op by default.

        Args:
            dataset: List of raw Observations from real hardware or sim rollouts.
        """
        pass

    def inverse(self, obs: Observation) -> Observation:
        """Invert this transform (optional).

        Useful for policy outputs that need to be un-normalized before being
        sent to a backend. Not all transforms are invertible — default raises.

        Args:
            obs: Transformed observation to invert.

        Returns:
            Observation in the original (pre-transform) space.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement inverse()."
        )


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------

class GaussianNoiseTransform(ITransform):
    """Add independent Gaussian noise to joint, velocity, and EE position fields.

    Use in sim training pipelines to mimic real encoder and sensor noise.
    Do NOT include in real deployment pipelines — real hardware already has noise.

    Args:
        joint_pos_std:  Noise std for joint positions (radians). 0 = disabled.
        joint_vel_std:  Noise std for joint velocities (rad/s). 0 = disabled.
        ee_pos_std:     Noise std for EE position (metres). 0 = disabled.
        seed:           Random seed. None = non-deterministic.
    """

    def __init__(
        self,
        joint_pos_std: float = 0.001,
        joint_vel_std: float = 0.005,
        ee_pos_std:    float = 0.001,
        seed:          int | None = None,
    ) -> None:
        self._joint_pos_std = joint_pos_std
        self._joint_vel_std = joint_vel_std
        self._ee_pos_std    = ee_pos_std
        self._rng           = np.random.default_rng(seed)

    def forward(self, obs: Observation) -> Observation:
        def _noise(arr: jnp.ndarray, std: float) -> jnp.ndarray:
            if std <= 0.0:
                return arr
            n = self._rng.normal(0.0, std, size=np.asarray(arr).shape).astype(np.float32)
            return jnp.array(np.asarray(arr) + n)

        return Observation(
            joint_positions     = _noise(obs.joint_positions, self._joint_pos_std),
            joint_velocities    = _noise(obs.joint_velocities, self._joint_vel_std),
            joint_torques       = obs.joint_torques,
            ee_position         = _noise(obs.ee_position, self._ee_pos_std),
            ee_orientation      = obs.ee_orientation,
            ee_velocity         = obs.ee_velocity,
            ee_angular_velocity = obs.ee_angular_velocity,
            rgb                 = obs.rgb,
            depth               = obs.depth,
            ft_force            = obs.ft_force,
            ft_torque           = obs.ft_torque,
            imu_acceleration    = obs.imu_acceleration,
            imu_angular_velocity= obs.imu_angular_velocity,
            gripper_state       = obs.gripper_state,
            timestamp           = obs.timestamp,
            timestamp_hw        = obs.timestamp_hw,
            timestamp_recv      = obs.timestamp_recv,
        )


class NormalizeTransform(ITransform):
    """Zero-mean / unit-variance normalization of joint and EE fields.

    Use the SAME NormalizeTransform instance (same statistics) in both sim
    and real pipelines. This is what ties the two distributions together.

    Three ways to construct:

    1. from_dataset(obs_list) — small datasets that fit in RAM:
           norm = NormalizeTransform.from_dataset(my_list_of_obs)

    2. from_stats(mean, std, ...) — production path, load pre-computed stats:
           stats = np.load("norm_stats.npz")
           norm  = NormalizeTransform.from_stats(**stats)

       Compute stats offline once on the full dataset without RAM constraints:
           norm = NormalizeTransform()
           for obs in stream:               # O(1) memory
               norm.fit_incremental(obs)
           np.savez("norm_stats.npz",
                    joint_pos_mean=norm.joint_pos_mean, ...)

    3. fit(dataset) — ITransform interface, same as from_dataset but in-place:
           norm = NormalizeTransform()
           norm.fit(dataset)

    Args:
        joint_pos_mean, joint_pos_std: Statistics for joint positions.
        joint_vel_mean, joint_vel_std: Statistics for joint velocities.
        ee_pos_mean, ee_pos_std:       Statistics for EE position.
    """

    def __init__(
        self,
        joint_pos_mean: Optional[np.ndarray] = None,
        joint_pos_std:  Optional[np.ndarray] = None,
        joint_vel_mean: Optional[np.ndarray] = None,
        joint_vel_std:  Optional[np.ndarray] = None,
        ee_pos_mean:    Optional[np.ndarray] = None,
        ee_pos_std:     Optional[np.ndarray] = None,
    ) -> None:
        self.joint_pos_mean = joint_pos_mean
        self.joint_pos_std  = joint_pos_std
        self.joint_vel_mean = joint_vel_mean
        self.joint_vel_std  = joint_vel_std
        self.ee_pos_mean    = ee_pos_mean
        self.ee_pos_std     = ee_pos_std
        # Welford accumulators (populated by fit_incremental)
        self._n: int = 0

    @classmethod
    def from_dataset(cls, dataset: list[Observation]) -> "NormalizeTransform":
        """Construct and fit from a list of Observations in one call.

        Loads all observations into RAM. Use fit_incremental() for large datasets.

        Args:
            dataset: Raw observations from real hardware or sim rollouts.

        Returns:
            Fitted NormalizeTransform ready to use in a pipeline.
        """
        instance = cls()
        instance.fit(dataset)
        return instance

    @classmethod
    def from_stats(
        cls,
        joint_pos_mean: np.ndarray,
        joint_pos_std:  np.ndarray,
        joint_vel_mean: np.ndarray,
        joint_vel_std:  np.ndarray,
        ee_pos_mean:    np.ndarray,
        ee_pos_std:     np.ndarray,
        **_ignored,
    ) -> "NormalizeTransform":
        """Construct from pre-computed statistics — the production path.

        Accepts **kwargs so np.load() output can be passed directly:
            stats = np.load("norm_stats.npz")
            norm  = NormalizeTransform.from_stats(**stats)

        Args:
            joint_pos_mean/std: Mean and std for joint positions [dof].
            joint_vel_mean/std: Mean and std for joint velocities [dof].
            ee_pos_mean/std:    Mean and std for EE position [3].

        Returns:
            NormalizeTransform with the given statistics, ready to use.
        """
        return cls(
            joint_pos_mean=np.asarray(joint_pos_mean, dtype=np.float32),
            joint_pos_std =np.asarray(joint_pos_std,  dtype=np.float32),
            joint_vel_mean=np.asarray(joint_vel_mean, dtype=np.float32),
            joint_vel_std =np.asarray(joint_vel_std,  dtype=np.float32),
            ee_pos_mean   =np.asarray(ee_pos_mean,    dtype=np.float32),
            ee_pos_std    =np.asarray(ee_pos_std,     dtype=np.float32),
        )

    def fit(self, dataset: list[Observation]) -> None:
        """Compute mean/std statistics in place from a dataset (ITransform interface).

        Loads all observations into RAM. For large datasets, prefer:
            for obs in stream: norm.fit_incremental(obs)
        """
        def _stats(arrs):
            stack = np.stack([np.asarray(a) for a in arrs])
            return stack.mean(0).astype(np.float32), (stack.std(0) + 1e-8).astype(np.float32)

        self.joint_pos_mean, self.joint_pos_std = _stats(
            [o.joint_positions  for o in dataset])
        self.joint_vel_mean, self.joint_vel_std = _stats(
            [o.joint_velocities for o in dataset])
        self.ee_pos_mean, self.ee_pos_std = _stats(
            [o.ee_position      for o in dataset])

    def fit_incremental(self, obs: Observation) -> None:
        """Update statistics from one observation using Welford's online algorithm.

        O(1) memory — call once per observation while streaming the dataset.
        After iterating all observations, save stats with np.savez() and load
        them at deployment time via from_stats().

        Args:
            obs: A single raw observation from hardware or sim.
        """
        jp = np.asarray(obs.joint_positions,  dtype=np.float64)
        jv = np.asarray(obs.joint_velocities, dtype=np.float64)
        ep = np.asarray(obs.ee_position,      dtype=np.float64)

        if self._n == 0:
            self._w_jp_mean = np.zeros_like(jp)
            self._w_jp_M2   = np.zeros_like(jp)
            self._w_jv_mean = np.zeros_like(jv)
            self._w_jv_M2   = np.zeros_like(jv)
            self._w_ep_mean = np.zeros_like(ep)
            self._w_ep_M2   = np.zeros_like(ep)

        self._n += 1
        n = self._n

        for val, mean, M2 in (
            (jp, self._w_jp_mean, self._w_jp_M2),
            (jv, self._w_jv_mean, self._w_jv_M2),
            (ep, self._w_ep_mean, self._w_ep_M2),
        ):
            delta  = val - mean
            mean  += delta / n         # in-place update (numpy array ref)
            M2    += delta * (val - mean)

        if n >= 2:
            self.joint_pos_mean = self._w_jp_mean.astype(np.float32)
            self.joint_pos_std  = (np.sqrt(self._w_jp_M2 / (n - 1)) + 1e-8).astype(np.float32)
            self.joint_vel_mean = self._w_jv_mean.astype(np.float32)
            self.joint_vel_std  = (np.sqrt(self._w_jv_M2 / (n - 1)) + 1e-8).astype(np.float32)
            self.ee_pos_mean    = self._w_ep_mean.astype(np.float32)
            self.ee_pos_std     = (np.sqrt(self._w_ep_M2 / (n - 1)) + 1e-8).astype(np.float32)

    def forward(self, obs: Observation) -> Observation:
        def _norm(arr, mean, std):
            if mean is None:
                return arr
            a = np.asarray(arr, dtype=np.float32)
            return jnp.array((a - mean) / std)

        return Observation(
            joint_positions     = _norm(obs.joint_positions, self.joint_pos_mean, self.joint_pos_std),
            joint_velocities    = _norm(obs.joint_velocities, self.joint_vel_mean, self.joint_vel_std),
            joint_torques       = obs.joint_torques,
            ee_position         = _norm(obs.ee_position, self.ee_pos_mean, self.ee_pos_std),
            ee_orientation      = obs.ee_orientation,
            ee_velocity         = obs.ee_velocity,
            ee_angular_velocity = obs.ee_angular_velocity,
            rgb                 = obs.rgb,
            depth               = obs.depth,
            ft_force            = obs.ft_force,
            ft_torque           = obs.ft_torque,
            imu_acceleration    = obs.imu_acceleration,
            imu_angular_velocity= obs.imu_angular_velocity,
            gripper_state       = obs.gripper_state,
            timestamp           = obs.timestamp,
            timestamp_hw        = obs.timestamp_hw,
            timestamp_recv      = obs.timestamp_recv,
        )

    def inverse(self, obs: Observation) -> Observation:
        """Undo normalization — useful for un-normalizing policy outputs."""
        def _denorm(arr, mean, std):
            if mean is None:
                return arr
            return jnp.array(np.asarray(arr, dtype=np.float32) * std + mean)

        return Observation(
            joint_positions     = _denorm(obs.joint_positions, self.joint_pos_mean, self.joint_pos_std),
            joint_velocities    = _denorm(obs.joint_velocities, self.joint_vel_mean, self.joint_vel_std),
            joint_torques       = obs.joint_torques,
            ee_position         = _denorm(obs.ee_position, self.ee_pos_mean, self.ee_pos_std),
            ee_orientation      = obs.ee_orientation,
            ee_velocity         = obs.ee_velocity,
            ee_angular_velocity = obs.ee_angular_velocity,
            rgb                 = obs.rgb,
            depth               = obs.depth,
            ft_force            = obs.ft_force,
            ft_torque           = obs.ft_torque,
            imu_acceleration    = obs.imu_acceleration,
            imu_angular_velocity= obs.imu_angular_velocity,
            gripper_state       = obs.gripper_state,
            timestamp           = obs.timestamp,
            timestamp_hw        = obs.timestamp_hw,
            timestamp_recv      = obs.timestamp_recv,
        )


class IdentityTransform(ITransform):
    """Pass-through. Useful as a placeholder in pipelines during development."""

    def forward(self, obs: Observation) -> Observation:
        return obs
