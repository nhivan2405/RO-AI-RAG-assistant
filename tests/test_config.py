import tempfile
import unittest
from pathlib import Path

from src.config import load_config


class ConfigTest(unittest.TestCase):
    def test_loads_defaults_for_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("chunking:\n  chunk_size: 256\n", encoding="utf-8")

            config = load_config(path)

        self.assertEqual(config["chunking"]["strategy"], "recursive")
        self.assertEqual(config["chunking"]["chunk_size"], 256)
        self.assertEqual(config["retrieval"]["top_k"], 5)
        self.assertEqual(config["vectordb"]["storage_path"], "qdrant_storage")
        self.assertEqual(config["vectordb"]["collection_name"], "rag_demo")

    def test_rejects_invalid_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("chunking:\n  chunk_size: 100\n  chunk_overlap: 100\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "chunking.chunk_overlap"):
                load_config(path)
