import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate import (
    _evaluation_storage_path,
    evaluate_queries,
    first_relevant_rank,
    load_questions,
    run_model_evaluation,
)


class FakeIndex:
    def __init__(self, responses):
        self.responses = responses

    def search(self, question, top_k):
        return self.responses[question][:top_k]


class ClosableFakeIndex(FakeIndex):
    def __init__(self, responses):
        super().__init__(responses)
        self.closed = False
        self.chunks = []

    def close(self):
        self.closed = True


class RetrievalEvaluationTest(unittest.TestCase):
    def test_anchor_format_remains_backward_compatible(self):
        hits = [
            {"chunk_id": "c0", "text": "Không liên quan"},
            {"chunk_id": "c1", "text": "Anchor đã xác minh"},
        ]

        self.assertEqual(first_relevant_rank(hits, "Anchor đã xác minh"), 2)

    def test_calculates_binary_relevance_metrics_for_multiple_chunk_labels(self):
        questions = [
            {
                "id": "multi-label",
                "type": "ground_truth",
                "category": "formulas",
                "question": "Câu hỏi",
                "relevant_chunk_ids": ["c1", "c3"],
            }
        ]
        index = FakeIndex(
            {
                "Câu hỏi": [
                    {"score": 0.9, "chunk_id": "c1", "text": "Một"},
                    {"score": 0.8, "chunk_id": "c2", "text": "Hai"},
                    {"score": 0.7, "chunk_id": "c3", "text": "Ba"},
                ]
            }
        )

        result = evaluate_queries(index, questions, top_k=3, min_score=0.5)

        metrics = result["metrics"]
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["precision_at_3"], 2 / 3)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertAlmostEqual(
            metrics["ndcg_at_3"], (1 + 0.5) / (1 + 1 / math.log2(3))
        )
        self.assertEqual(metrics["hit_at_3"], 1.0)

    def test_reports_out_of_scope_decisions_and_empty_categories(self):
        questions = [
            {
                "id": "outside",
                "type": "out_of_scope",
                "category": "out_of_scope",
                "question": "Ngoài phạm vi",
            }
        ]
        index = FakeIndex(
            {"Ngoài phạm vi": [{"score": 0.49, "chunk_id": "c0", "text": "Khác"}]}
        )

        result = evaluate_queries(index, questions, top_k=5, min_score=0.5)

        self.assertEqual(result["by_category"]["out_of_scope"]["rejected_count"], 1)
        self.assertEqual(result["by_category"]["tables"]["query_count"], 0)
        self.assertEqual(result["by_category"]["tables"]["metrics"], {})

    def test_rejects_ground_truth_without_any_relevance_label(self):
        dataset = [
            {
                "id": "missing-label",
                "type": "ground_truth",
                "category": "tables",
                "question": "Thiếu nhãn",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected_anchor or relevant_chunk_ids"):
                load_questions(path)

    def test_uses_dedicated_storage_for_each_model(self):
        root = Path("results/qdrant_evaluation")

        minilm_path = _evaluation_storage_path(
            root, "run-01", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        bge_path = _evaluation_storage_path(root, "run-01", "BAAI/bge-m3")

        self.assertNotEqual(minilm_path, bge_path)
        self.assertTrue(str(minilm_path).startswith(str(root)))
        self.assertTrue(str(bge_path).startswith(str(root)))

    def test_closes_evaluation_index_after_run(self):
        index = ClosableFakeIndex({"Câu hỏi": []})
        config = {"retrieval": {"top_k": 5, "min_score": 0.5}, "chunking": {}}
        questions = [
            {
                "id": "out-of-scope",
                "type": "out_of_scope",
                "category": "out_of_scope",
                "question": "Câu hỏi",
            }
        ]
        with patch("evaluate.build_evaluation_index", return_value=(index, config, 384)):
            result = run_model_evaluation(
                "data",
                "model/test",
                "evaluation_collection",
                Path("results/qdrant_evaluation/run/model"),
                questions,
            )

        self.assertTrue(index.closed)
        self.assertEqual(
            result["run_configuration"]["storage_path"],
            "results\\qdrant_evaluation\\run\\model",
        )

    def test_closes_evaluation_index_when_evaluation_fails(self):
        index = ClosableFakeIndex({})
        config = {"retrieval": {"top_k": 5, "min_score": 0.5}, "chunking": {}}
        with patch("evaluate.build_evaluation_index", return_value=(index, config, 384)):
            with patch("evaluate.evaluate_queries", side_effect=RuntimeError("failed")):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    run_model_evaluation(
                        "data",
                        "model/test",
                        "evaluation_collection",
                        Path("results/qdrant_evaluation/run/model"),
                        [],
                    )

        self.assertTrue(index.closed)
