import sys
import unittest
from unittest.mock import MagicMock, patch

import main as application
from src.retrieval.retriever import FALLBACK_MESSAGE, Retriever


class RetrieverTest(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever(min_score=0.5)

    def test_is_answerable_rejects_empty_hits(self):
        self.assertFalse(self.retriever.is_answerable([]))

    def test_is_answerable_rejects_score_below_threshold(self):
        self.assertFalse(self.retriever.is_answerable([{"score": 0.49}]))

    def test_is_answerable_accepts_score_equal_to_threshold(self):
        self.assertTrue(self.retriever.is_answerable([{"score": 0.5}]))

    def test_is_answerable_accepts_score_above_threshold(self):
        self.assertTrue(self.retriever.is_answerable([{"score": 0.51}]))

    @patch("main.answer_with_gemini")
    @patch("main.RagIndex")
    @patch("main.chunk_documents")
    @patch("main.load_documents")
    @patch("main.load_config")
    @patch("main.load_dotenv")
    def test_rejected_query_does_not_call_gemini(
        self,
        load_dotenv_mock,
        load_config_mock,
        load_documents_mock,
        chunk_documents_mock,
        rag_index_class,
        answer_with_gemini_mock,
    ):
        self._configure_cli_mocks(
            load_config_mock,
            rag_index_class,
            hits=[{"score": 0.49, "text": "Yếu", "source": "a.txt", "page": 1}],
        )

        with patch.object(sys, "argv", ["main.py"]), patch("builtins.print") as print_mock:
            application.main()

        answer_with_gemini_mock.assert_not_called()
        self.assertIn(FALLBACK_MESSAGE, str(print_mock.call_args_list))

    @patch("main.answer_with_gemini", return_value="Câu trả lời có căn cứ")
    @patch("main.RagIndex")
    @patch("main.chunk_documents")
    @patch("main.load_documents")
    @patch("main.load_config")
    @patch("main.load_dotenv")
    def test_accepted_hits_keep_order_and_metadata_for_gemini(
        self,
        load_dotenv_mock,
        load_config_mock,
        load_documents_mock,
        chunk_documents_mock,
        rag_index_class,
        answer_with_gemini_mock,
    ):
        hits = [
            {
                "score": 0.8,
                "text": "Hit đầu tiên",
                "source": "first.txt",
                "page": 1,
                "chunk_id": "first.txt:p1:c0",
            },
            {
                "score": 0.7,
                "text": "Hit thứ hai",
                "source": "second.txt",
                "page": 2,
                "chunk_id": "second.txt:p2:c0",
            },
        ]
        self._configure_cli_mocks(load_config_mock, rag_index_class, hits=hits)

        with patch.object(sys, "argv", ["main.py"]), patch("builtins.print"):
            application.main()

        answer_with_gemini_mock.assert_called_once_with(
            "Sinh viên được mượn bao nhiêu quyển sách?", hits
        )

    @patch("main.answer_with_gemini")
    @patch("main.RagIndex")
    @patch("main.chunk_documents")
    @patch("main.load_documents")
    @patch("main.load_config")
    @patch("main.load_dotenv")
    def test_no_llm_rejection_prints_diagnostic_and_fallback(
        self,
        load_dotenv_mock,
        load_config_mock,
        load_documents_mock,
        chunk_documents_mock,
        rag_index_class,
        answer_with_gemini_mock,
    ):
        self._configure_cli_mocks(
            load_config_mock,
            rag_index_class,
            hits=[{"score": 0.49, "text": "Yếu", "source": "a.txt", "page": 1}],
        )

        with patch.object(sys, "argv", ["main.py", "--no-llm"]), patch(
            "builtins.print"
        ) as print_mock:
            application.main()

        output = str(print_mock.call_args_list)
        answer_with_gemini_mock.assert_not_called()
        self.assertIn("Retrieval decision: REJECTED", output)
        self.assertIn("top_score=0.4900", output)
        self.assertIn("min_score=0.5000", output)
        self.assertIn(FALLBACK_MESSAGE, output)

    @patch("main.answer_with_gemini")
    @patch("main.RagIndex")
    @patch("main.chunk_documents")
    @patch("main.load_documents")
    @patch("main.load_config")
    @patch("main.load_dotenv")
    def test_no_llm_empty_hits_prints_safe_rejection(
        self,
        load_dotenv_mock,
        load_config_mock,
        load_documents_mock,
        chunk_documents_mock,
        rag_index_class,
        answer_with_gemini_mock,
    ):
        self._configure_cli_mocks(load_config_mock, rag_index_class, hits=[])

        with patch.object(sys, "argv", ["main.py", "--no-llm"]), patch(
            "builtins.print"
        ) as print_mock:
            application.main()

        output = str(print_mock.call_args_list)
        answer_with_gemini_mock.assert_not_called()
        self.assertIn("Retrieval decision: REJECTED", output)
        self.assertIn("top_score=N/A", output)
        self.assertIn(FALLBACK_MESSAGE, output)

    @patch("main.answer_with_gemini")
    @patch("main.RagIndex")
    @patch("main.chunk_documents")
    @patch("main.load_documents")
    @patch("main.load_config")
    @patch("main.load_dotenv")
    def test_no_llm_acceptance_keeps_raw_hits_and_prints_diagnostic(
        self,
        load_dotenv_mock,
        load_config_mock,
        load_documents_mock,
        chunk_documents_mock,
        rag_index_class,
        answer_with_gemini_mock,
    ):
        self._configure_cli_mocks(
            load_config_mock,
            rag_index_class,
            hits=[{"score": 0.8, "text": "Đủ", "source": "a.txt", "page": 1}],
        )

        with patch.object(sys, "argv", ["main.py", "--no-llm"]), patch(
            "builtins.print"
        ) as print_mock:
            application.main()

        output = str(print_mock.call_args_list)
        answer_with_gemini_mock.assert_not_called()
        self.assertIn("[1] score=0.8000 source=a.txt page=1 | Đủ", output)
        self.assertIn("Retrieval decision: ACCEPTED", output)
        self.assertIn("top_score=0.8000", output)

    @staticmethod
    def _configure_cli_mocks(load_config_mock, rag_index_class, hits):
        load_config_mock.return_value = {
            "chunking": {
                "chunk_size": 700,
                "chunk_overlap": 100,
                "strategy": "recursive",
            },
            "retrieval": {"top_k": 5, "min_score": 0.5},
        }
        index = rag_index_class.return_value
        index.model_name = "mock-model"
        index.build.return_value = 384
        index.search.return_value = hits
