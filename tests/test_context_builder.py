import unittest
import tempfile
from pathlib import Path

from src.chunking.chunker import Chunk
from src.prompts.context_builder import build_context
from src.vectordb.vector_store import PersistentVectorStore


# Kiểm tra evidence nguyên vẹn và citation mapping từ hit thật.
class ContextBuilderTest(unittest.TestCase):
    def test_orders_evidence_by_score_and_maps_citations(self):
        hits = [
            {"score": 0.7, "text": "Evidence B", "source": "b.txt", "chunk_id": "b"},
            {
                "score": 0.9,
                "text": "Evidence A",
                "source": "a.txt",
                "page": 2,
                "section": "# A",
                "chunk_id": "a",
            },
        ]

        context = build_context(hits, token_budget=20)

        self.assertEqual(context.text, "[1] Evidence A\n\n[2] Evidence B")
        self.assertEqual(context.citations[0].metadata["page"], 2)
        self.assertEqual(context.citations[1].metadata, {"source": "b.txt", "chunk_id": "b"})

    def test_deduplicates_chunk_ids_and_preserves_first_ranked_evidence(self):
        hits = [
            {"score": 0.9, "text": "Preferred", "chunk_id": "same"},
            {"score": 0.8, "text": "Duplicate", "chunk_id": "same"},
        ]

        context = build_context(hits, token_budget=10)

        self.assertEqual(context.text, "[1] Preferred")
        self.assertEqual(len(context.citations), 1)

    def test_skips_oversized_evidence_without_truncating_citation(self):
        hits = [
            {"score": 0.9, "text": "one two three four", "chunk_id": "large"},
            {"score": 0.8, "text": "small", "chunk_id": "small"},
        ]

        context = build_context(hits, token_budget=3)

        self.assertEqual(context.text, "[1] small")
        self.assertEqual(context.token_count, 2)
        self.assertEqual(context.citations[0].metadata, {"chunk_id": "small"})

    def test_returns_empty_context_for_invalid_evidence(self):
        hits = [
            {"score": 0.9, "text": ""},
            {"score": "invalid", "text": "Evidence"},
            {"score": 0.8},
        ]

        context = build_context(hits, token_budget=10)

        self.assertEqual(context.text, "")
        self.assertEqual(context.citations, ())
        self.assertEqual(context.token_count, 0)

    def test_rejects_non_positive_token_budget(self):
        with self.assertRaisesRegex(ValueError, "token_budget"):
            build_context([], token_budget=0)

    def test_accepts_actual_vector_store_search_hit_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PersistentVectorStore(
                storage_path=Path(directory),
                collection_name="context_hits",
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                vector_dimension=384,
            )
            chunks = [
                Chunk(
                    "Nội dung có section.",
                    "guide.txt",
                    2,
                    "guide.txt:p2:c0",
                    section="# Network",
                ),
                Chunk("Nội dung thiếu section.", "guide.txt", None, "guide.txt:pNone:c1"),
            ]
            vectors = [[1.0] + [0.0] * 383, [0.5] + [0.0] * 383]
            store.index_chunks(chunks, vectors)
            hits = store.search([1.0] + [0.0] * 383, top_k=2)
            store.close()

        context = build_context(hits, token_budget=20)

        self.assertEqual([citation.citation_id for citation in context.citations], [1, 2])
        self.assertEqual(context.citations[0].metadata["source"], "guide.txt")
        self.assertEqual(context.citations[0].metadata["page"], 2)
        self.assertEqual(context.citations[0].metadata["section"], "# Network")
        self.assertNotIn("page", context.citations[1].metadata)
        self.assertNotIn("section", context.citations[1].metadata)
