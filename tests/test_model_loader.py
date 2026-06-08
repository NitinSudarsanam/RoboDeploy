from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.policies.learned.loader import ModelContractError, ModelLoader


class ModelLoaderTests(unittest.TestCase):
    def test_load_custom_predict_fn(self):
        def predict_fn(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
            return np.asarray(obs_dict["state"][:3], dtype=np.float64)

        loader = ModelLoader(predict_fn=predict_fn)
        model = loader.load(
            {
                "framework": "custom",
                "checkpoint": "unused.pt",
                "expected_action_space": ActionSpace.JOINT_POS,
                "expected_action_dim": 3,
                "expected_obs_keys": ["state"],
            }
        )
        out = model.predict_fn({"state": np.ones(5, dtype=np.float32)})
        self.assertEqual(out.shape, (3,))

    def test_contract_error_on_action_dim(self):
        loader = ModelLoader(
            predict_fn=lambda obs: np.zeros(3),
        )
        with self.assertRaises(ModelContractError):
            loader.load(
                {
                    "framework": "custom",
                    "checkpoint": "x.pt",
                    "expected_action_dim": 7,
                }
            )

    def test_resolve_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "policy.pt"
            ckpt.write_bytes(b"stub")
            loader = ModelLoader(search_paths=[root])
            resolved = loader.resolve("policy.pt")
            self.assertEqual(resolved, ckpt)

    def test_hf_download_mocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            target = cache / "openvla__openvla-7b" / "model.pt"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"hf")

            loader = ModelLoader(hf_cache=cache)
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(target)) as dl:
                resolved = loader.resolve("hf://openvla/openvla-7b/model.pt")
            self.assertEqual(resolved, target)
            dl.assert_called_once()


    def test_openvla_import_error_without_package(self):
        loader = ModelLoader()
        with mock.patch.object(ModelLoader, "resolve", return_value=Path("openvla.pt")):
            with self.assertRaises(ImportError) as ctx:
                loader.load(
                    {
                        "framework": "openvla",
                        "checkpoint": "hf://openvla/openvla-7b/model.pt",
                        "expected_action_dim": 7,
                    }
                )
        self.assertIn("openvla", str(ctx.exception).lower())

    def test_pi0_import_error_without_package(self):
        loader = ModelLoader()
        with mock.patch.object(ModelLoader, "resolve", return_value=Path("pi0.pt")):
            with self.assertRaises(ImportError) as ctx:
                loader.load(
                    {
                        "framework": "pi0",
                        "checkpoint": "hf://physical-intelligence/pi0/model.pt",
                        "expected_action_dim": 7,
                    }
                )
        self.assertIn("pi0", str(ctx.exception).lower())

    def test_octo_import_error_without_package(self):
        loader = ModelLoader()
        with mock.patch.object(ModelLoader, "resolve", return_value=Path("octo.pt")):
            with self.assertRaises(ImportError) as ctx:
                loader.load(
                    {
                        "framework": "octo",
                        "checkpoint": "hf://rail-berkeley/octo-base/model.pt",
                        "expected_action_dim": 7,
                    }
                )
        self.assertIn("octo", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
