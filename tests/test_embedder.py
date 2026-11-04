import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.embeddings.embedder import Embedder, MODEL_DIMENSIONS


class EmbedderTest(unittest.TestCase):
    def test_embed_texts_for_supported_models(self):
        for model_name, dimension in MODEL_DIMENSIONS.items():
            with self.subTest(model_name=model_name):
                model = MagicMock()
                model.encode.return_value = np.ones((2, dimension))

                with patch(
                    "src.embeddings.embedder.SentenceTransformer",
                    return_value=model,
                ):
                    embedder = Embedder(model_name)
                    vectors = embedder.embed_texts(["tài liệu một", "tài liệu hai"])

                self.assertEqual(len(vectors), 2)
                self.assertEqual(len(vectors[0]), dimension)
                self.assertEqual(embedder.dimension, dimension)
                model.encode.assert_called_once_with(
                    ["tài liệu một", "tài liệu hai"],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

    def test_embed_query_for_supported_models(self):
        for model_name, dimension in MODEL_DIMENSIONS.items():
            with self.subTest(model_name=model_name):
                model = MagicMock()
                model.encode.return_value = np.zeros((1, dimension))

                with patch(
                    "src.embeddings.embedder.SentenceTransformer",
                    return_value=model,
                ):
                    embedder = Embedder(model_name)
                    vector = embedder.embed_query("câu hỏi kiểm thử")

                self.assertEqual(len(vector), dimension)
                self.assertEqual(embedder.dimension, dimension)
                model.encode.assert_called_once_with(
                    ["câu hỏi kiểm thử"],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

    def test_rejects_model_with_unexpected_dimension(self):
        model = MagicMock()
        model.encode.return_value = np.zeros((1, 1024))

        with patch(
            "src.embeddings.embedder.SentenceTransformer",
            return_value=model,
        ):
            embedder = Embedder(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            with self.assertRaisesRegex(ValueError, "returned dimension 1024"):
                embedder.embed_query("câu hỏi")

    def test_rejects_unsupported_model(self):
        with self.assertRaisesRegex(ValueError, "Unsupported embedding model"):
            Embedder("unsupported/model")
