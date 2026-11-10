import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.vectordb.connection import create_qdrant_client


# Kiểm tra chọn backend và đóng client khi readiness thất bại.
class QdrantConnectionTest(unittest.TestCase):
    @patch("src.vectordb.connection.QdrantClient")
    def test_creates_local_client_with_persistent_path(self, client_class):
        with tempfile.TemporaryDirectory() as directory:
            create_qdrant_client("local", Path(directory) / "storage")

        client_class.assert_called_once_with(path=str(Path(directory) / "storage"))

    @patch("src.vectordb.connection.QdrantClient")
    def test_creates_docker_client_and_checks_readiness(self, client_class):
        client = client_class.return_value

        result = create_qdrant_client(
            "docker",
            "ignored-local-path",
            url="http://localhost:6333",
            api_key="test-key",
        )

        self.assertIs(result, client)
        client_class.assert_called_once_with(
            url="http://localhost:6333", api_key="test-key"
        )
        client.get_collections.assert_called_once_with()

    @patch("src.vectordb.connection.QdrantClient")
    def test_closes_docker_client_when_readiness_fails(self, client_class):
        client = client_class.return_value
        client.get_collections.side_effect = OSError("offline")

        with self.assertRaisesRegex(RuntimeError, "Docker server is unavailable"):
            create_qdrant_client("docker", "ignored", url="http://localhost:6333")

        client.close.assert_called_once_with()

    @patch("src.vectordb.connection.package_version", return_value="1.20.0")
    @patch("src.vectordb.connection.QdrantClient")
    def test_creates_cloud_client_with_https_key_and_compatible_server(
        self, client_class, package_version_mock
    ):
        client = client_class.return_value
        client.info.return_value = SimpleNamespace(version="1.14.3")

        result = create_qdrant_client(
            "cloud",
            "ignored-local-path",
            url="https://cluster.example.cloud.qdrant.io",
            api_key="test-key",
        )

        self.assertIs(result, client)
        client.get_collections.assert_called_once_with()
        client.info.assert_called_once_with()
        package_version_mock.assert_called_once_with("qdrant-client")

    def test_rejects_cloud_without_https_or_api_key(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            create_qdrant_client("cloud", "ignored", url="http://localhost:6333", api_key="key")
        with self.assertRaisesRegex(RuntimeError, "QDRANT_API_KEY"):
            create_qdrant_client("cloud", "ignored", url="https://cluster.example", api_key=None)

    @patch("src.vectordb.connection.package_version", return_value="2.0.0")
    @patch("src.vectordb.connection.QdrantClient")
    def test_closes_cloud_client_when_server_version_is_incompatible(
        self, client_class, package_version_mock
    ):
        client = client_class.return_value
        client.info.return_value = SimpleNamespace(version="1.14.3")

        with self.assertRaisesRegex(RuntimeError, "unavailable or incompatible"):
            create_qdrant_client(
                "cloud",
                "ignored",
                url="https://cluster.example.cloud.qdrant.io",
                api_key="test-key",
            )

        client.close.assert_called_once_with()
        package_version_mock.assert_called_once_with("qdrant-client")

    def test_rejects_unsupported_backend(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            create_qdrant_client("invalid", "ignored")
