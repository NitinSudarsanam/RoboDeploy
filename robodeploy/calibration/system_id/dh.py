"""DH parameter identification (delegates to nonlinear kinematic fit)."""

from __future__ import annotations

import numpy as np

from robodeploy.calibration.kinematic.nonlinear import DHParams, NonlinearDHCalibration


class DHEstimator:
    """Fit small DH perturbations from FK/IK pose pairs."""

    def __init__(self, n_links: int = 6) -> None:
        self._cal = NonlinearDHCalibration(n_links=n_links)

    def fit(self, pose_pairs: list[tuple[np.ndarray, np.ndarray]]) -> DHParams:
        return self._cal.fit(pose_pairs)
