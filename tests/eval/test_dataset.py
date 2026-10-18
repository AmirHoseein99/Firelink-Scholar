from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eval.dataset import (
    DatasetError,
    build_dataset,
    configure_judge,
    judge_configured,
    load_dataset,
    validate_dataset,
)
from eval.storage import save_json


class TestDataset(unittest.TestCase):
    def test_build_leaves_judge_unconfigured(self) -> None:
        dataset = build_dataset()
        self.assertFalse(judge_configured(dataset))
        self.assertEqual(dataset["judge"]["model"], "")
        validate_dataset(dataset)

    def test_configure_judge_pins_without_grading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.json"
            dataset = build_dataset()
            save_json(path, dataset)
            updated = configure_judge(
                load_dataset(path),
                model="judge-model",
                backend="freellm",
                path=path,
            )
            self.assertTrue(judge_configured(updated))
            self.assertEqual(updated["judge"]["model"], "judge-model")
            self.assertEqual(load_dataset(path)["judge"]["model"], "judge-model")

    def test_configured_judge_requires_model(self) -> None:
        dataset = build_dataset()
        dataset["judge"]["configured"] = True
        with self.assertRaises(DatasetError):
            validate_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
