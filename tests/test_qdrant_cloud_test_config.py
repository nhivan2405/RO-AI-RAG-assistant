import unittest

from tests.test_qdrant_cloud_integration import _resolve_cloud_credentials


# Bảo vệ ưu tiên biến test trước credential chuẩn từ môi trường.
class QdrantCloudTestConfigurationTest(unittest.TestCase):
    def test_test_variables_override_standard_environment_variables(self):
        url, api_key = _resolve_cloud_credentials(
            {
                "QDRANT_CLOUD_TEST_URL": "https://test.cloud.qdrant.io",
                "QDRANT_CLOUD_TEST_API_KEY": "test-key",
                "QDRANT_URL": "https://default.cloud.qdrant.io",
                "QDRANT_API_KEY": "default-key",
            }
        )

        self.assertEqual(url, "https://test.cloud.qdrant.io")
        self.assertEqual(api_key, "test-key")

    def test_standard_environment_variables_are_used_as_fallback(self):
        url, api_key = _resolve_cloud_credentials(
            {
                "QDRANT_URL": "https://default.cloud.qdrant.io",
                "QDRANT_API_KEY": "default-key",
            }
        )

        self.assertEqual(url, "https://default.cloud.qdrant.io")
        self.assertEqual(api_key, "default-key")
