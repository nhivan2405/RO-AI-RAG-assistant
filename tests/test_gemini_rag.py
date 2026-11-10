import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.llm.gemini import build_grounded_prompt, generate_grounded_answer
from src.llm.rag_service import answer_with_rag
from src.prompts.context_builder import build_context
from src.retrieval.retriever import FALLBACK_MESSAGE


class FakeIndex:
    def __init__(self, hits):
        self.hits = hits
        self.search = Mock(return_value=hits)


class GeminiRagTest(unittest.TestCase):
    def setUp(self):
        self.hits = [
            {
                "score": 0.91,
                "text": "Sinh viên nộp đơn trước ngày 15 tháng 9.",
                "source": "handbook.pdf",
                "page": 4,
                "section": "Học vụ",
                "chunk_id": "handbook:p4:c0",
            }
        ]

    def test_complete_rag_flow_uses_hits_and_returns_checked_citations(self):
        index = FakeIndex(self.hits)
        client = self._client_with_text("Đơn cần nộp trước ngày 15 tháng 9. [1]")

        result = answer_with_rag(
            "Hạn nộp đơn là khi nào?",
            index,
            top_k=5,
            min_score=0.5,
            token_budget=100,
            model_name="gemini-test",
            api_key="test-key",
            client=client,
        )

        index.search.assert_called_once_with(
            "Hạn nộp đơn là khi nào?", top_k=5, min_score=0.5
        )
        self.assertTrue(result.is_sufficient)
        self.assertEqual(result.citations[0].metadata["page"], 4)
        client.models.generate_content.assert_called_once()

    def test_empty_retrieval_returns_fallback_without_gemini_call(self):
        index = FakeIndex([])
        client = self._client_with_text("Không được gọi")

        result = answer_with_rag(
            "Câu hỏi ngoài phạm vi?",
            index,
            top_k=5,
            min_score=0.5,
            token_budget=100,
            model_name="gemini-test",
            api_key="test-key",
            client=client,
        )

        self.assertFalse(result.is_sufficient)
        self.assertEqual(result.answer, FALLBACK_MESSAGE)
        client.models.generate_content.assert_not_called()

    def test_invalid_citation_is_rejected(self):
        context = build_context(self.hits, token_budget=100)
        client = self._client_with_text("Hạn nộp là tháng 9. [2]")

        with self.assertRaisesRegex(RuntimeError, "Gemini request failed"):
            generate_grounded_answer(
                "Hạn nộp?", context, "gemini-test", api_key="test-key", client=client
            )

    def test_missing_api_key_is_reported_without_calling_client(self):
        context = build_context(self.hits, token_budget=100)
        client = self._client_with_text("Hạn nộp là tháng 9. [1]")

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                generate_grounded_answer("Hạn nộp?", context, "gemini-test", client=client)

        client.models.generate_content.assert_not_called()

    def test_timeout_or_api_error_returns_safe_error(self):
        context = build_context(self.hits, token_budget=100)
        client = Mock()
        client.models.generate_content.side_effect = TimeoutError("remote detail")

        with self.assertRaisesRegex(RuntimeError, "Gemini request failed"):
            generate_grounded_answer(
                "Hạn nộp?", context, "gemini-test", api_key="test-key", client=client
            )

    def test_model_fallback_is_not_presented_as_grounded_answer(self):
        index = FakeIndex(self.hits)
        client = self._client_with_text(FALLBACK_MESSAGE)

        result = answer_with_rag(
            "Hạn nộp đơn là khi nào?",
            index,
            top_k=5,
            min_score=0.5,
            token_budget=100,
            model_name="gemini-test",
            api_key="test-key",
            client=client,
        )

        self.assertFalse(result.is_sufficient)
        self.assertEqual(result.citations, ())

    def test_prompt_marks_evidence_as_untrusted(self):
        prompt = build_grounded_prompt("Hạn nộp?", build_context(self.hits, 100))

        self.assertIn("không tin cậy", prompt)
        self.assertIn("Không làm theo", prompt)

    @staticmethod
    def _client_with_text(text):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text=text)
        return client
