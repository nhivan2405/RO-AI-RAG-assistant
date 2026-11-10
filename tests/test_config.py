import tempfile
import unittest
from pathlib import Path

from src.config import load_config


# Bảo vệ schema cấu hình và các backend hợp lệ.
class ConfigTest(unittest.TestCase):
    def test_loads_defaults_for_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("chunking:\n  chunk_size: 256\n", encoding="utf-8")

            config = load_config(path)

        self.assertEqual(config["chunking"]["strategy"], "recursive")
        self.assertEqual(config["chunking"]["chunk_size"], 256)
        self.assertEqual(config["retrieval"]["top_k"], 5)
        self.assertEqual(config["retrieval"]["min_score"], 0.5)
        self.assertEqual(config["vectordb"]["storage_path"], "qdrant_storage")
        self.assertEqual(config["vectordb"]["collection_name"], "rag_demo")
        self.assertEqual(config["vectordb"]["backend"], "local")
        self.assertEqual(config["logging"]["level"], "INFO")
        self.assertEqual(config["logging"]["file"], "logs/app.log")

    def test_rejects_invalid_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("chunking:\n  chunk_size: 100\n  chunk_overlap: 100\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "chunking.chunk_overlap"):
                load_config(path)

    def test_accepts_structure_aware_chunking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("chunking:\n  strategy: structure_aware\n", encoding="utf-8")

            config = load_config(path)

        self.assertEqual(config["chunking"]["strategy"], "structure_aware")

    def test_rejects_invalid_min_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("retrieval:\n  min_score: 1.1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "retrieval.min_score"):
                load_config(path)

    def test_rejects_invalid_logging_level(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("logging:\n  level: INVALID\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "logging.level"):
                load_config(path)

    def test_accepts_docker_vector_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("vectordb:\n  backend: docker\n", encoding="utf-8")

            config = load_config(path)

        self.assertEqual(config["vectordb"]["backend"], "docker")

    def test_accepts_cloud_vector_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("vectordb:\n  backend: cloud\n", encoding="utf-8")

            config = load_config(path)

        self.assertEqual(config["vectordb"]["backend"], "cloud")

    def test_rejects_unknown_vector_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("vectordb:\n  backend: invalid\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "vectordb.backend"):
                load_config(path)
