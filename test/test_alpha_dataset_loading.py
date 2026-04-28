import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from alpha_model import utils


class AlphaDatasetLoadingTests(unittest.TestCase):
    def test_resolve_streaming_mode_auto_for_large_datasets(self):
        self.assertTrue(utils._resolve_streaming_mode("math_instruct", "auto"))
        self.assertTrue(utils._resolve_streaming_mode("metamath", "auto"))
        self.assertTrue(utils._resolve_streaming_mode("magicoder", "auto"))
        self.assertFalse(utils._resolve_streaming_mode("gsm8k", "auto"))

    def test_load_dataset_falls_back_to_local_math_instruct_mirror(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / "datasets"
            local_dir.mkdir()
            local_file = local_dir / "MathInstruct.json"
            local_file.write_text('[{"instruction": "Solve x + 1 = 2"}]', encoding="utf-8")

            kwargs, _ = utils._build_load_kwargs("math_instruct", None, False)
            env_updates = {
                "ALPHA_DATASET_DIR": str(local_dir),
                "HF_HUB_OFFLINE": "",
                "HF_DATASETS_OFFLINE": "",
            }
            calls = []

            def fake_load_dataset(**candidate_kwargs):
                calls.append(candidate_kwargs)
                if candidate_kwargs["path"] == "TIGER-Lab/MathInstruct":
                    raise ConnectionError("primary hub load failed")
                return {"loaded_from": candidate_kwargs["data_files"]["train"]}

            with mock.patch.dict(os.environ, env_updates, clear=False):
                with mock.patch.object(utils, "load_dataset", side_effect=fake_load_dataset):
                    dataset = utils._load_dataset_with_fallback(kwargs, "math_instruct")

            self.assertEqual(dataset, {"loaded_from": str(local_file)})
            self.assertEqual(calls[0]["path"], "TIGER-Lab/MathInstruct")
            self.assertEqual(calls[1]["path"], "json")
            self.assertEqual(calls[1]["data_files"]["train"], str(local_file))

    def test_offline_error_mentions_expected_metamath_mirror(self):
        kwargs, _ = utils._build_load_kwargs("metamath", None, False)

        def fake_load_dataset(**candidate_kwargs):
            raise ConnectionError(f"cannot load {candidate_kwargs['path']}")

        env_updates = {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "",
            "ALPHA_DATASET_DIR": "",
            "HF_DATASET_MIRROR_DIR": "",
        }
        with mock.patch.dict(os.environ, env_updates, clear=False):
            with mock.patch.object(utils, "load_dataset", side_effect=fake_load_dataset):
                with self.assertRaises(RuntimeError) as exc_info:
                    utils._load_dataset_with_fallback(kwargs, "metamath")

        message = str(exc_info.exception)
        self.assertIn("metamath", message)
        self.assertIn("MetaMathQA-395K.json", message)
        self.assertIn("HF_HUB_OFFLINE", message)


if __name__ == "__main__":
    unittest.main()
