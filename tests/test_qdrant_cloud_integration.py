import os
import tempfile
import unittest
import uuid
from pathlib import Path

from dotenv import load_dotenv

from src.chunking.chunker import Chunk
from src.vectordb.vector_store import PersistentVectorStore


def _load_cloud_test_credentials() -> tuple[str | None, str | None]:
    load_dotenv(override=False)
    return _resolve_cloud_credentials(os.environ)


def _resolve_cloud_credentials(environ: dict[str, str]) -> tuple[str | None, str | None]:
    url = environ.get("QDRANT_CLOUD_TEST_URL") or environ.get("QDRANT_URL")
    api_key = environ.get("QDRANT_CLOUD_TEST_API_KEY") or environ.get("QDRANT_API_KEY")
    return url, api_key


def _has_cloud_credentials(url: str | None, api_key: str | None) -> bool:
    return bool(
        url
        and url.startswith("https://")
        and api_key
        and api_key != "YOUR_QDRANT_API_KEY_HERE"
    )


CLOUD_URL, CLOUD_API_KEY = _load_cloud_test_credentials()


@unittest.skipUnless(
    _has_cloud_credentials(CLOUD_URL, CLOUD_API_KEY),
    "Set QDRANT_CLOUD_TEST_URL and QDRANT_CLOUD_TEST_API_KEY to run Cloud integration",
)
# Collection ngẫu nhiên cô lập kiểm tra persistence mà không dùng dữ liệu học tập.
class QdrantCloudIntegrationTest(unittest.TestCase):
    def test_indexes_searches_and_persists_in_isolated_collection(self):
        collection_name = f"ro_ai_cloud_test_{uuid.uuid4().hex}"
        vector = [1.0, 0.0, 0.0, 0.0]
        chunk = Chunk("Public Cloud test content", "cloud-test.txt", 1, "cloud:p1:c0")
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
            storage_path="unused-by-cloud",
            metadata_path=Path(directory),
            collection_name=collection_name,
            model_name="cloud-test-model",
            vector_dimension=4,
            backend="cloud",
            url=CLOUD_URL,
            api_key=CLOUD_API_KEY,
        )
