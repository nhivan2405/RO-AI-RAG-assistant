import unittest
from unittest.mock import MagicMock, patch

from src.chunking.chunker import Chunk
from src.pipeline import RagIndex


MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RagIndexCompatibilityTest(unittest.TestCase):
    @patch("src.pipeline.PersistentVectorStore")
    @patch("src.pipeline.Embedder")
    @patch("src.pipeline.load_config")
    def test_build_and_search_use_extracted_modules(
        self,
        load_config_mock,
        embedder_class,
        vector_store_class,
    ):
        load_config_mock.return_value = {
            "models": {"embedding": MINILM_MODEL, "gemini": "gemini-2.5-flash"},
            "vectordb": {
                "storage_path": "qdrant_storage",
                "collection_name": "rag_demo",
            },
        }
        embedder = embedder_class.return_value
        embedder.expected_dimension = 384
        embedder.dimension = 384
        embedder.embed_texts.return_value = [[0.1] * 384]
        embedder.embed_query.return_value = [0.2] * 384
        vector_store = vector_store_class.return_value
        vector_store.vector_dimension = 384
        vector_store.client = MagicMock()
        vector_store.search.return_value = [{"score": 0.9, "text": "Nội dung"}]
        chunks = [Chunk("Nội dung", "guide.txt", 1, "guide.txt:p1:c0")]

        index = RagIndex()
        dimension = index.build(chunks)
        hits = index.search("Câu hỏi", top_k=3)

        self.assertEqual(index.model_name, MINILM_MODEL)
        self.assertEqual(index.collection, "rag_demo")
        self.assertEqual(dimension, 384)
        self.assertEqual(hits, [{"score": 0.9, "text": "Nội dung"}])
        embedder.embed_texts.assert_called_once_with(["Nội dung"])
        vector_store.index_chunks.assert_called_once_with(chunks, [[0.1] * 384])
        vector_store.search.assert_called_once_with([0.2] * 384, 3)
