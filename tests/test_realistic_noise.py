from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.transforms import (
    BiasDriftTransform,
    ColoredNoiseTransform,
    DropoutTransform,
    LatencyTransform,
    QuantizationTransform,
)
from robodeploy.tasks.action_noise import ActionNoiseInjector, ExternalDisturbanceInjector
from robodeploy.core.types import Action
from robodeploy.testing import make_obs


class RealisticNoiseTests(unittest.TestCase):
    def test_colored_noise_ou_changes_obs(self):
        tx = ColoredNoiseTransform(kind="ou", sigma=0.01, dt=0.02, seed=0)
        o0 = make_obs(0.0)
        o1 = tx.forward(o0)
        self.assertNotEqual(float(o1.joint_positions[0]), float(o0.joint_positions[0]))

    def test_quantization_snaps_to_ticks(self):
        tx = QuantizationTransform(ticks_per_unit={"joint_positions": 10.0})
        o = make_obs(0.123)
        out = tx.forward(o)
        val = float(out.joint_positions[0])
        self.assertAlmostEqual(val, round(0.123 * 10) / 10, places=5)

    def test_bias_drift_accumulates(self):
        tx = BiasDriftTransform(drift_rate=1e-3, max_drift=0.05, seed=0)
        o = make_obs(0.0)
        vals = []
        for _ in range(50):
            o = tx.forward(o)
            vals.append(float(o.joint_positions[0]))
        self.assertGreater(max(vals) - min(vals), 0.0)

    def test_dropout_holds_stale_frame(self):
        tx = DropoutTransform(p=1.0, max_stale_steps=3, seed=0)
        o0 = make_obs(0.0)
        o1 = make_obs(1.0)
        tx.forward(o0)
        held = tx.forward(o1)
        self.assertEqual(float(held.joint_positions[0]), 0.0)

    def test_latency_transform_two_steps(self):
        tx = LatencyTransform(latency_steps=2, seed=0)
        for v in (0.0, 1.0, 2.0, 3.0):
            tx.forward(make_obs(v))
        out = tx.forward(make_obs(4.0))
        self.assertEqual(float(out.joint_positions[0]), 2.0)

    def test_action_noise_injector(self):
        inj = ActionNoiseInjector(joint_noise_std=0.01, seed=0)
        a = Action(joint_positions=np.array([0.5, 0.5], dtype=np.float32))
        out = inj(a)
        self.assertIsNotNone(out.joint_positions)

    def test_external_disturbance_noop_without_backend_hook(self):
        inj = ExternalDisturbanceInjector(probability_per_step=1.0, seed=0)

        class _Backend:
            pass

        inj.inject(_Backend())  # should not raise


if __name__ == "__main__":
    unittest.main()
