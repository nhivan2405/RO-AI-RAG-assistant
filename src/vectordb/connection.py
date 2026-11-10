"""Tạo Qdrant client theo backend đã chọn."""
from __future__ import annotations

from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from src.utils.helpers import get_logger


logger = get_logger("qdrant_connection")


def create_qdrant_client(
    backend: str,
    storage_path: str | Path,
    url: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    """Tạo client Local, Docker hoặc Cloud theo backend đã chọn."""
    if backend == "local":
        path = Path(storage_path)
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    if backend == "docker":
        return _create_docker_client(url, api_key)
    if backend == "cloud":
        return _create_cloud_client(url, api_key)
    raise ValueError("Qdrant backend must be 'local', 'docker', or 'cloud'")


def _create_docker_client(url: str | None, api_key: str | None) -> QdrantClient:
    if not url or not url.strip():
        raise ValueError("Qdrant Docker URL must be configured")
    client = QdrantClient(url=url, api_key=api_key)
    try:
        client.get_collections()
    except Exception as error:
        client.close()
        logger.error("qdrant_docker_unavailable error_type=%s", type(error).__name__)
        raise RuntimeError("Qdrant Docker server is unavailable") from error
    return client


def _create_cloud_client(url: str | None, api_key: str | None) -> QdrantClient:
    _validate_cloud_credentials(url, api_key)
    client = QdrantClient(url=url, api_key=api_key)
    try:
        # Readiness trước khi tạo collection để không ghi vào endpoint lỗi.
        client.get_collections()
        _validate_server_version(client)
    except Exception as error:
        client.close()
        logger.error("qdrant_cloud_unavailable error_type=%s", type(error).__name__)
        raise RuntimeError("Qdrant Cloud server is unavailable or incompatible") from error
    return client


def _validate_cloud_credentials(url: str | None, api_key: str | None) -> None:
    if not url or not url.startswith("https://"):
        raise ValueError("Qdrant Cloud URL must use HTTPS")
    if not api_key or api_key == "YOUR_QDRANT_API_KEY_HERE":
        raise RuntimeError("Set QDRANT_API_KEY before using Qdrant Cloud")


def _validate_server_version(client: Any) -> None:
    client_major = _version_major(package_version("qdrant-client"))
    server_major = _version_major(str(client.info().version))
    if client_major != server_major:
        raise RuntimeError("Qdrant client and server major versions are incompatible")
    logger.info(
        "qdrant_cloud_version_compatible client_major=%s server_major=%s",
        client_major,
        server_major,
    )


def _version_major(value: str) -> int:
    first_part = value.split(".", maxsplit=1)[0]
    if not first_part.isdigit():
        raise ValueError("Qdrant version must start with a numeric major version")
    return int(first_part)
