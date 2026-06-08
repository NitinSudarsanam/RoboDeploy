from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkReproducibilityTests(unittest.TestCase):
    def test_same_seed_same_scores(self):
        from robodeploy.evaluation.runner import run_eval

        kwargs = dict(
            benchmark="manipulation_v1/reach_target",
            policy="scripted",
            backend="dummy",
            episodes=6,
            base_seed=123,
            benchmarks_root=str(REPO_ROOT / "benchmarks"),
        )
        a = run_eval(**kwargs)
        b = run_eval(**kwargs)
        self.assertEqual(len(a.episodes), len(b.episodes))
        for ep_a, ep_b in zip(a.episodes, b.episodes):
            self.assertEqual(ep_a.success, ep_b.success)
            self.assertAlmostEqual(ep_a.reward_total, ep_b.reward_total, places=6)
            self.assertEqual(ep_a.steps, ep_b.steps)

    def test_subproc_parallel_reproducible(self):
        from robodeploy.evaluation.runner import run_eval

        kwargs = dict(
            benchmark="manipulation_v1/pick_place_cube",
            policy="scripted",
            backend="dummy",
            episodes=6,
            base_seed=7,
            benchmarks_root=str(REPO_ROOT / "benchmarks"),
        )
        seq = run_eval(**kwargs, parallel=False)
        par = run_eval(**kwargs, parallel=True, n_workers=3)
        self.assertEqual(seq.aggregate.n_episodes, par.aggregate.n_episodes)
        self.assertAlmostEqual(seq.aggregate.success_rate, par.aggregate.success_rate, places=5)
        self.assertAlmostEqual(seq.aggregate.mean_reward, par.aggregate.mean_reward, places=5)


if __name__ == "__main__":
    unittest.main()
