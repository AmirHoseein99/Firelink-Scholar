from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eval.storage import load_json, save_json, sha256_json


class TestStorage(unittest.TestCase):
    def test_atomic_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "data.json"
            save_json(path, {"b": 1, "a": [1, 2]})
            self.assertEqual(load_json(path), {"a": [1, 2], "b": 1})
            self.assertEqual(load_json(Path(tmp) / "missing.json", default={}), {})

    def test_canonical_hash_is_order_independent(self) -> None:
        self.assertEqual(
            sha256_json({"a": 1, "b": 2}), sha256_json({"b": 2, "a": 1})
        )


if __name__ == "__main__":
    unittest.main()
