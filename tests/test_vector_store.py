import tempfile
import unittest
from pathlib import Path

from src.chunking.chunker import Chunk
from src.vectordb.vector_store import PersistentVectorStore


MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BGE_M3_MODEL = "BAAI/bge-m3"


class PersistentVectorStoreTest(unittest.TestCase):
    def test_creates_collection_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._create_store(directory)
            chunks = self._chunks()
            inserted = store.index_chunks(chunks, self._vectors(len(chunks)))
            hits = store.search([0.1] * 384, top_k=2)
            store.close()

        self.assertEqual(inserted, 2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(
            set(hits[0]),
            {
                "score",
                "text",
                "source",
                "page",
                "chunk_id",
                "section",
                "split_reason",
                "content_hash",
            },
        )
        self.assertEqual(hits[0]["source"], "guide.txt")

    def test_reuses_collection_and_skips_unchanged_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._create_store(directory)
            chunks = self._chunks()
            store.index_chunks(chunks, self._vectors(len(chunks)))
            store.close()

            reopened_store = self._create_store(directory)
            inserted = reopened_store.index_chunks(chunks, self._vectors(len(chunks)))
            hits = reopened_store.search([0.1] * 384, top_k=2)
            reopened_store.close()

        self.assertEqual(inserted, 0)
        self.assertEqual(len(hits), 2)

    def test_persists_collection_vectors_and_metadata_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            vector = [1.0] + [0.0] * 383
            chunk = Chunk(
                "Nội dung cần tồn tại sau khi khởi động lại",
                "restart.txt",
                4,
                "restart.txt:p4:c0",
                section="# Network",
                split_reason="oversized_table",
            )
            store = self._create_store(directory)
            store.index_chunks([chunk], [vector])
            store.close()

            reopened_store = self._create_store(directory)
            self.assertTrue(
                reopened_store.client.collection_exists("library_chunks")
            )
            raw_result = reopened_store.client.query_points(
                collection_name="library_chunks",
                query=vector,
                limit=1,
                with_payload=True,
                with_vectors=True,
            )
            hits = reopened_store.search(vector, top_k=1)
            reopened_store.close()

        persisted_point = raw_result.points[0]
        self.assertEqual(list(persisted_point.vector), vector)
        self.assertEqual(persisted_point.payload["text"], chunk.text)
        self.assertEqual(persisted_point.payload["source"], "restart.txt")
        self.assertEqual(persisted_point.payload["page"], 4)
        self.assertEqual(persisted_point.payload["chunk_id"], "restart.txt:p4:c0")
        self.assertEqual(persisted_point.payload["section"], "# Network")
        self.assertEqual(persisted_point.payload["split_reason"], "oversized_table")
        self.assertEqual(hits[0]["chunk_id"], "restart.txt:p4:c0")
        self.assertEqual(hits[0]["source"], "restart.txt")

    def test_rejects_model_mismatch_for_existing_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._create_store(directory)
            store.close()

            with self.assertRaisesRegex(ValueError, "incompatible with model"):
                PersistentVectorStore(
                    storage_path=directory,
                    collection_name="library_chunks",
                    model_name=BGE_M3_MODEL,
                    vector_dimension=1024,
                )

    def test_rejects_vectors_with_wrong_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._create_store(directory)
            with self.assertRaisesRegex(ValueError, "Vector dimension 3 does not match 384"):
                store.index_chunks(self._chunks()[:1], [[0.1, 0.2, 0.3]])
            store.close()

    @staticmethod
    def _create_store(directory: str) -> PersistentVectorStore:
        return PersistentVectorStore(
            storage_path=Path(directory),
            collection_name="library_chunks",
            model_name=MINILM_MODEL,
            vector_dimension=384,
        )

    @staticmethod
    def _chunks() -> list[Chunk]:
        return [
            Chunk("Nội dung một", "guide.txt", 1, "guide.txt:p1:c0"),
            Chunk("Nội dung hai", "guide.txt", 1, "guide.txt:p1:c1"),
        ]

    @staticmethod
    def _vectors(count: int) -> list[list[float]]:
        return [[0.1] * 384 for _ in range(count)]
