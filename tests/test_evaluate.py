import unittest

from evaluate import evaluate_queries, first_relevant_rank


class FakeIndex:
    def __init__(self, responses):
        self.responses = responses

    def search(self, question, top_k):
        return self.responses[question][:top_k]


class RetrievalEvaluationTest(unittest.TestCase):
    def test_finds_first_chunk_containing_ground_truth_anchor(self):
        hits = [
            {"chunk_id": "sample.txt:pNone:c0", "text": "Không liên quan"},
            {"chunk_id": "sample.txt:pNone:c1", "text": "Anchor đã xác minh"},
        ]
        self.assertEqual(first_relevant_rank(hits, "Anchor đã xác minh"), 2)

    def test_calculates_metrics_and_out_of_scope_decisions(self):
        questions = [
            {
                "id": "rank-two",
                "type": "ground_truth",
                "question": "Một",
                "expected_anchor": "Anchor một",
            },
            {
                "id": "miss",
                "type": "ground_truth",
                "question": "Hai",
                "expected_anchor": "Anchor hai",
            },
            {"id": "accepted", "type": "out_of_scope", "question": "Ba"},
            {"id": "rejected", "type": "out_of_scope", "question": "Bốn"},
        ]
        index = FakeIndex({
            "Một": [
                {"score": 0.9, "chunk_id": "c0", "text": "Khác"},
                {"score": 0.8, "chunk_id": "c1", "text": "Anchor một"},
            ],
            "Hai": [{"score": 0.8, "chunk_id": "c2", "text": "Khác"}],
            "Ba": [{"score": 0.5, "chunk_id": "c3", "text": "Khác"}],
            "Bốn": [{"score": 0.49, "chunk_id": "c4", "text": "Khác"}],
        })

        result = evaluate_queries(index, questions, top_k=5, min_score=0.5)

        self.assertEqual(result["metrics"]["hit_at_5"], 0.5)
        self.assertEqual(result["metrics"]["mrr"], 0.25)
        self.assertEqual(
            [item["decision"] for item in result["out_of_scope_results"]],
            ["ACCEPTED", "REJECTED"],
        )
