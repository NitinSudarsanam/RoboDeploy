"""
ObsPipeline — ordered composition of ITransform objects.

ObsPipeline no longer knows whether it is running in sim or on real hardware.
The distinction is entirely in which transforms you include:

    # Sim pipeline — add noise to match real hardware, then normalize
    sim_pipeline = ObsPipeline([
        GaussianNoiseTransform(joint_pos_std=0.001, ee_pos_std=0.001),
        NormalizeTransform.from_dataset(dataset),
    ])

    # Real pipeline — only normalize (real hardware has real noise already)
    real_pipeline = ObsPipeline([
        NormalizeTransform.from_dataset(dataset),   # exact same statistics as sim
    ])

Both pipelines call process(obs) identically. No `is_real` flag. No branching.
The execution path is identical — only the transform list differs.

This resolves the leaky abstraction: if you see `if is_real:` inside a pipeline,
that is a sign the two deployment contexts are diverging, which is the primary
cause of sim-to-real failure in production systems.

RoboEnv accepts the pipeline at construction time. For quick prototyping, pass
no pipeline and get a no-op IdentityTransform pipeline by default.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

from robodeploy.core.transforms import ITransform, IdentityTransform
from robodeploy.core.types import Observation, SensorData


class ObsSyncMode(str, Enum):
    DROP_LATEST = "drop_latest"
    TIME_WINDOW = "time_window"


class SensorSampleBuffer:
    """Per-sensor latest samples merged when timestamps align within a window."""

    def __init__(self, window_s: float = 0.05) -> None:
        self.window_s = float(window_s)
        self._latest: dict[str, SensorData] = {}

    def push(self, name: str, data: SensorData) -> None:
        self._latest[str(name)] = data

    def merge(self, obs: Observation) -> Observation:
        if not self._latest:
            return obs
        anchor = float(obs.timestamp_hw or obs.timestamp)
        images = dict(obs.images)
        depths = dict(obs.depths)
        rgb = obs.rgb
        depth = obs.depth
        ft_force = obs.ft_force
        ft_torque = obs.ft_torque
        objects = dict(getattr(obs, "objects", {}) or {})
        sensor_status = dict(getattr(obs, "sensor_status", {}) or {})
        camera_frames = dict(getattr(obs, "camera_frames", {}) or {})
        camera_intrinsics = dict(getattr(obs, "camera_intrinsics", {}) or {})
        for name, sample in self._latest.items():
            ts = float(sample.timestamp_hw or sample.timestamp)
            if abs(ts - anchor) > self.window_s:
                continue
            if getattr(sample, "status", "ok") != "ok":
                sensor_status[name] = str(sample.status)
            if sample.rgb is not None:
                images[name] = sample.rgb
                if rgb is None:
                    rgb = sample.rgb
            if sample.depth is not None:
                depths[name] = sample.depth
                if depth is None:
                    depth = sample.depth
            ft_force = sample.ft_force if sample.ft_force is not None else ft_force
            ft_torque = sample.ft_torque if sample.ft_torque is not None else ft_torque
            if getattr(sample, "objects", None):
                objects.update(sample.objects)
            if getattr(sample, "frame_id", None):
                camera_frames[name] = str(sample.frame_id)
            if getattr(sample, "intrinsics", None):
                camera_intrinsics[name] = dict(sample.intrinsics)
        return replace(
            obs,
            rgb=rgb,
            depth=depth,
            images=images,
            depths=depths,
            ft_force=ft_force,
            ft_torque=ft_torque,
            objects=objects,
            sensor_status=sensor_status,
            camera_frames=camera_frames,
            camera_intrinsics=camera_intrinsics,
        )

    def reset(self) -> None:
        self._latest.clear()


class ObsPipeline:
    """Applies an ordered list of ITransforms to each observation.

    Construct with a list of transforms appropriate for your deployment context
    (sim or real). Pass the same NormalizeTransform instance in both contexts
    to guarantee matching statistics.

    Args:
        transforms: Ordered list of transforms. Applied left-to-right.
                    Defaults to [IdentityTransform()] — a no-op for prototyping.

    Example:
        # Sim (training)
        norm = NormalizeTransform.fit(real_dataset)
        sim_pipeline = ObsPipeline([GaussianNoiseTransform(), norm])

        # Real (deployment) — same norm instance, no noise
        real_pipeline = ObsPipeline([norm])

        # Both pipelines used identically:
        obs = pipeline.process(raw_obs)
    """

    def __init__(
        self,
        transforms: list[ITransform] | None = None,
        *,
        sync_mode: ObsSyncMode = ObsSyncMode.DROP_LATEST,
        sync_window_s: float = 0.05,
    ) -> None:
        self.transforms: list[ITransform] = transforms or [IdentityTransform()]
        self.sync_mode: ObsSyncMode = sync_mode
        self.sync_window_s = float(sync_window_s)
        self._latest_sync_obs: Observation | None = None
        self._last_processed: Observation | None = None
        self._sensor_buffer = SensorSampleBuffer(sync_window_s)

    def buffer_sensor(self, name: str, data: SensorData) -> None:
        """Store an out-of-band sensor read for timestamp-aligned merge in process()."""
        self._sensor_buffer.push(name, data)

    @staticmethod
    def with_primary_fields(obs: Observation) -> Observation:
        """Mirror legacy rgb/depth from named sensor dicts when unset."""
        rgb = obs.rgb
        depth = obs.depth
        if rgb is None and obs.images:
            rgb = next(iter(obs.images.values()))
        if depth is None and obs.depths:
            depth = next(iter(obs.depths.values()))
        if rgb is obs.rgb and depth is obs.depth:
            return obs
        return replace(obs, rgb=rgb, depth=depth)

    def process(self, obs: Observation) -> Observation:
        """Apply all transforms in order and return the result.

        Called by RoboEnv on every step. Must be fast.

        Args:
            obs: Raw observation from the backend.

        Returns:
            Transformed observation ready for the policy.
        """
        if not self.sync_policy(obs):
            if self._last_processed is not None:
                return self._last_processed
        obs = self._sensor_buffer.merge(obs)
        obs = self.with_primary_fields(obs)
        for transform in self.transforms:
            obs = transform.forward(obs)
        self._last_processed = obs
        return obs

    def reset_sync(self) -> None:
        """Clear sync/process buffers at episode boundaries."""
        self._latest_sync_obs = None
        self._last_processed = None
        self._sensor_buffer.reset()

    def fit(self, dataset: list[Observation]) -> None:
        """Fit all stateful transforms (e.g. NormalizeTransform) from a dataset.

        Calls fit() on every transform in the pipeline. Stateless transforms
        (GaussianNoiseTransform, IdentityTransform) ignore this call.

        Args:
            dataset: List of raw Observations from real hardware or sim rollouts.
        """
        for transform in self.transforms:
            transform.fit(dataset)

    def append(self, transform: ITransform) -> "ObsPipeline":
        """Add a transform to the end of the pipeline. Returns self for chaining.

        Args:
            transform: Transform to append.

        Returns:
            self, so you can chain: pipeline.append(t1).append(t2)
        """
        self.transforms.append(transform)
        return self

    def sync_policy(self, obs: Observation) -> bool:
        """Return whether an observation is within the configured sync policy.

        ARCHITECTURE.md describes strategies like DROP_LATEST and TIME_WINDOW.
        """
        if self.sync_mode == ObsSyncMode.DROP_LATEST:
            self._latest_sync_obs = obs
            return True
        if self._latest_sync_obs is None:
            self._latest_sync_obs = obs
            return True
        latest_hw = float(self._latest_sync_obs.timestamp_hw or self._latest_sync_obs.timestamp)
        current_hw = float(obs.timestamp_hw or obs.timestamp)
        if abs(current_hw - latest_hw) <= self.sync_window_s:
            self._latest_sync_obs = obs
            return True
        return False

    def __repr__(self) -> str:
        names = [type(t).__name__ for t in self.transforms]
        return f"ObsPipeline([{', '.join(names)}])"
