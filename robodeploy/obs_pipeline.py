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

from enum import Enum

from robodeploy.core.transforms import ITransform, IdentityTransform
from robodeploy.core.types      import Observation


class ObsSyncMode(str, Enum):
    DROP_LATEST = "drop_latest"
    TIME_WINDOW = "time_window"


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

    def process(self, obs: Observation) -> Observation:
        """Apply all transforms in order and return the result.

        Called by RoboEnv on every step. Must be fast.

        Args:
            obs: Raw observation from the backend.

        Returns:
            Transformed observation ready for the policy.
        """
        for transform in self.transforms:
            obs = transform.forward(obs)
        return obs

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
