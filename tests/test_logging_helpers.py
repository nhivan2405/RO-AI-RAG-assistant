import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.embeddings.embedder import Embedder
from src.ingestion.loader import load_documents
from src.pipeline import answer_with_gemini
from src.utils.helpers import APP_LOGGER_NAME, setup_logging
from src.vectordb.vector_store import PersistentVectorStore


# Đảm bảo log chẩn đoán không vô tình ghi câu hỏi hoặc khóa bí mật.
class LoggingHelpersTest(unittest.TestCase):
    def tearDown(self):
        self._close_log_handlers()

    @staticmethod
    def _close_log_handlers():
        logger = logging.getLogger(APP_LOGGER_NAME)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def test_setup_logging_creates_file_without_duplicate_handlers(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "nested" / "app.log"
            config = {"logging": {"level": "INFO", "file": str(log_path)}}

            logger = setup_logging(config)
            logger.info("safe_event event_count=1")
            repeated_logger = setup_logging(config)
            repeated_logger.info("safe_event_after_repeat event_count=1")
            for handler in logger.handlers:
                handler.flush()

            content = log_path.read_text(encoding="utf-8")
            handler_count = len(logger.handlers)
            self._close_log_handlers()

        self.assertIs(logger, repeated_logger)
        self.assertEqual(handler_count, 1)
        self.assertIn("safe_event event_count=1", content)
        self.assertEqual(content.count("safe_event event_count=1"), 1)
        self.assertEqual(content.count("safe_event_after_repeat event_count=1"), 1)

    def test_failure_logs_exclude_raw_question_and_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "app.log"
            setup_logging({"logging": {"level": "INFO", "file": str(log_path)}})
            raw_question = "CÂU_HỎI_BÍ_MẬT"
            fake_key = "API_KEY_BÍ_MẬT"

            with patch("src.llm.gemini.os.getenv", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                    answer_with_gemini(raw_question, [])

            content = log_path.read_text(encoding="utf-8")
            self._close_log_handlers()

        self.assertIn("gemini_api_key_missing", content)
        self.assertNotIn(raw_question, content)
        self.assertNotIn(fake_key, content)

    def test_document_failure_logs_without_folder_name(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "app.log"
            setup_logging({"logging": {"level": "INFO", "file": str(log_path)}})
            missing_folder = Path(directory) / "TÀI_LIỆU_BÍ_MẬT"

            with self.assertRaises(FileNotFoundError):
                load_documents(str(missing_folder))

            content = log_path.read_text(encoding="utf-8")
            self._close_log_handlers()

        self.assertIn("document_folder_missing", content)
        self.assertNotIn(missing_folder.name, content)

    def test_embedding_initialization_failure_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "app.log"
            setup_logging({"logging": {"level": "INFO", "file": str(log_path)}})

            with patch(
                "src.embeddings.embedder.SentenceTransformer",
                side_effect=RuntimeError("model unavailable"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Unable to initialize"):
                    Embedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

            content = log_path.read_text(encoding="utf-8")
            self._close_log_handlers()

        self.assertIn("embedding_initialization_failed", content)
        self.assertNotIn("model unavailable", content)

    def test_qdrant_initialization_failure_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "app.log"
            setup_logging({"logging": {"level": "INFO", "file": str(log_path)}})

            with patch(
                "src.vectordb.connection.QdrantClient",
                side_effect=OSError("storage unavailable"),
            ):
                with self.assertRaises(OSError):
                    PersistentVectorStore(
                        storage_path=Path(directory) / "store",
                        collection_name="test_collection",
                        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        vector_dimension=384,
                    )

            content = log_path.read_text(encoding="utf-8")
            self._close_log_handlers()

        self.assertIn("qdrant_initialization_failed", content)
        self.assertNotIn("storage unavailable", content)

    def test_gemini_api_failure_is_logged_without_question_or_key(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "app.log"
            setup_logging({"logging": {"level": "INFO", "file": str(log_path)}})
            raw_question = "CÂU_HỎI_KHÔNG_ĐƯỢC_LOG"
            fake_key = "KHÓA_KHÔNG_ĐƯỢC_LOG"

            with patch("src.pipeline.load_config") as load_config_mock, patch(
                "src.pipeline.generate_grounded_answer",
                side_effect=RuntimeError("Gemini request failed"),
            ):
                load_config_mock.return_value = {"models": {"gemini": "gemini-test"}}
                with self.assertRaisesRegex(RuntimeError, "Gemini request failed"):
                    answer_with_gemini(
                        raw_question,
                        [{"score": 0.9, "text": "Evidence", "chunk_id": "test:c0"}],
                    )

            content = log_path.read_text(encoding="utf-8")
            self._close_log_handlers()

        self.assertIn("legacy_gemini_request_failed", content)
        self.assertNotIn(raw_question, content)
        self.assertNotIn(fake_key, content)
