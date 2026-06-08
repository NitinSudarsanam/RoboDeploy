from __future__ import annotations

import unittest
from pathlib import Path

from robodeploy.sim2real.config import (
    Sim2RealPair,
    apply_shared_fields,
    load_pair_for_preset,
    load_sim2real_pair,
    merge_preset_with_dr,
    pair_name_from_preset,
    resolve_sim2real_pair,
)
from robodeploy.tasks.randomization import DomainRandomizerConfig, RandomLevel


class Sim2RealConfigTests(unittest.TestCase):
    def test_load_kuka_pair_from_yaml(self):
        pairs_file = Path(__file__).resolve().parents[1] / "examples" / "config" / "sim2real_pairs.yaml"
        pair = load_sim2real_pair("kuka_pick", pairs_file=pairs_file)
        self.assertEqual(pair.sim_preset, "kuka_pick_mujoco")
        self.assertEqual(pair.real_preset, "kuka_sensor_pick_mujoco")
        self.assertIn("custom_modules", pair.shared)

    def test_resolve_inline_pair_dict(self):
        pair = resolve_sim2real_pair(
            {"sim_preset": "a_sim", "real_preset": "a_real", "shared": {"policy": "p"}}
        )
        self.assertEqual(pair.sim_preset, "a_sim")
        self.assertEqual(pair.shared["policy"], "p")

    def test_apply_shared_fields_merges_without_clobber(self):
        pair = Sim2RealPair(
            name="t",
            sim_preset="s",
            real_preset="r",
            shared={"obs_spec_policy": "raise", "task_kwargs": {"foo": 1}},
        )
        merged = apply_shared_fields({"robot": "kuka", "task_kwargs": {"bar": 2}}, pair)
        self.assertEqual(merged["robot"], "kuka")
        self.assertEqual(merged["obs_spec_policy"], "raise")
        self.assertEqual(merged["task_kwargs"], {"foo": 1, "bar": 2})

    def test_merge_preset_with_dr_attaches_task_kwargs(self):
        preset = {"robot": "kuka", "backend": "mujoco", "task": "pick", "policy": "p"}
        dr = DomainRandomizerConfig(level=RandomLevel.FULL, seed=3)
        merged = merge_preset_with_dr(preset, dr)
        dr_block = merged["task_kwargs"]["domain_randomization"]
        self.assertEqual(dr_block["level"], "FULL")
        self.assertEqual(dr_block["seed"], 3)

    def test_benchmark_preset_pairs_resolve(self):
        pairs_file = Path(__file__).resolve().parents[1] / "examples" / "config" / "sim2real_pairs.yaml"
        preset = {"sim2real_pair": "kuka_reach_dummy", "backend": "dummy"}
        self.assertEqual(pair_name_from_preset(preset), "kuka_reach_dummy")
        pair = load_pair_for_preset(preset, pairs_file=pairs_file)
        assert pair is not None
        self.assertEqual(pair.sim_preset, "kuka_pick_mujoco")
        self.assertEqual(pair.real_preset, "kuka_sensor_pick_mujoco")

    def test_all_sim2real_benchmark_pair_names_exist(self):
        pairs_file = Path(__file__).resolve().parents[1] / "examples" / "config" / "sim2real_pairs.yaml"
        expected = {
            "kuka_reach_dummy",
            "kuka_reach_mujoco",
            "kuka_reach_real",
            "kuka_pick_dummy",
            "kuka_pick_mujoco",
            "kuka_pick_real",
            "kuka_peg_dummy",
            "kuka_peg_mujoco",
            "kuka_peg_real",
        }
        for name in sorted(expected):
            pair = load_sim2real_pair(name, pairs_file=pairs_file)
            self.assertTrue(pair.sim_preset)
            self.assertTrue(pair.real_preset)


if __name__ == "__main__":
    unittest.main()
