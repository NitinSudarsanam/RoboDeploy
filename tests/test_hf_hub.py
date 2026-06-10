from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.policies.learned.hf_hub import HFModelRegistry
from robodeploy.policies.learned.loader import LoadedModel, ModelLoader


class HFModelRegistryTests(unittest.TestCase):
    def test_from_name_mocked_e2e(self):
        def predict_fn(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
            return np.zeros(7, dtype=np.float64)

        loaded = LoadedModel(
            predict_fn=predict_fn,
            predict_batch_fn=None,
            action_space=ActionSpace.DELTA_EE,
            action_dim=7,
            required_obs_keys=["rgb", "instruction"],
            framework="openvla",
            metadata={},
        )
        loader = ModelLoader(predict_fn=predict_fn)
        with mock.patch.object(ModelLoader, "load", return_value=loaded) as load_mock:
            policy = HFModelRegistry.from_name(
                "openvla-7b",
                action_space=ActionSpace.DELTA_EE,
                loader=loader,
            )
        load_mock.assert_called_once()
        spec = load_mock.call_args[0][0]
        self.assertEqual(spec["framework"], "openvla")
        self.assertEqual(policy.action_space, ActionSpace.DELTA_EE)
        self.assertEqual(policy._model.action_dim, 7)

    def test_download_resolves_hf_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            target = cache / "openvla__openvla-7b" / "model.pt"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"hf")

            loader = ModelLoader(hf_cache=cache)
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(target)):
                path = HFModelRegistry.download("openvla-7b", loader=loader)
            self.assertEqual(path, str(target))


if __name__ == "__main__":
    unittest.main()
