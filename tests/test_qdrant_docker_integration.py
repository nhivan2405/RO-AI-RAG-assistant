import os
import tempfile
import unittest
import uuid
from pathlib import Path

from src.chunking.chunker import Chunk
from src.vectordb.vector_store import PersistentVectorStore


DOCKER_URL = os.getenv("QDRANT_DOCKER_TEST_URL")


@unittest.skipUnless(DOCKER_URL, "Set QDRANT_DOCKER_TEST_URL to run Docker integration")
# Chỉ chạy khi endpoint Docker được cấp rõ ràng để tránh chạm production.
class QdrantDockerIntegrationTest(unittest.TestCase):
    def test_indexes_searches_and_persists(self):
        collection_name = f"qdrant_docker_test_{uuid.uuid4().hex}"
        vector = [1.0, 0.0, 0.0, 0.0]
        chunk = Chunk("Quy định kiểm thử Docker", "handbook.txt", 2, "docker:p2:c0")
        with tempfile.TemporaryDirectory() as directory:
            store = self._create_store(directory, collection_name)
            try:
                self.assertEqual(store.index_chunks([chunk], [vector]), 1)
                self.assertEqual(store.search(vector, top_k=1)[0]["chunk_id"], chunk.chunk_id)
                store.close()

                reopened = self._create_store(directory, collection_name)
                try:
                    self.assertEqual(reopened.search(vector, top_k=1)[0]["source"], chunk.source)
                finally:
                    reopened.client.delete_collection(collection_name)
                    reopened.close()
            except Exception:
                store.close()
                raise

    @staticmethod
    def _create_store(directory: str, collection_name: str) -> PersistentVectorStore:
        return PersistentVectorStore(
            storage_path="unused-by-docker",
            metadata_path=Path(directory),
            collection_name=collection_name,
            model_name="docker-test-model",
            vector_dimension=4,
            backend="docker",
            url=DOCKER_URL,
        )
