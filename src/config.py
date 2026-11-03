"""Đọc và kiểm tra cấu hình không nhạy cảm của ứng dụng."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "chunking": {
        "strategy": "recursive",
        "chunk_size": 700,
        "chunk_overlap": 100,
    },
    "retrieval": {"top_k": 5},
    "models": {
        "embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "gemini": "gemini-2.5-flash",
    },
}


def load_config(config_path: str | Path = "config.yaml") -> dict[str, dict[str, Any]]:
    """Trả về cấu hình đã được hợp nhất với giá trị mặc định."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        loaded_config = yaml.safe_load(config_file) or {}

    if not isinstance(loaded_config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    config = deepcopy(DEFAULT_CONFIG)
    for section, values in loaded_config.items():
        if section not in config:
            raise ValueError(f"Unknown configuration section: {section}")
        if not isinstance(values, dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping")
        unknown_keys = values.keys() - config[section].keys()
        if unknown_keys:
            names = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unknown configuration keys in '{section}': {names}")
        config[section].update(values)

    _validate_config(config)
    return config


def _validate_config(config: dict[str, dict[str, Any]]) -> None:
    chunking = config["chunking"]
    strategy = chunking["strategy"]
    if strategy not in {"fixed", "recursive"}:
        raise ValueError("chunking.strategy must be 'fixed' or 'recursive'")

    chunk_size = chunking["chunk_size"]
    chunk_overlap = chunking["chunk_overlap"]
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunking.chunk_size must be a positive integer")
    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise ValueError("chunking.chunk_overlap must be an integer")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunking.chunk_overlap must satisfy 0 <= overlap < chunk_size")

    top_k = config["retrieval"]["top_k"]
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("retrieval.top_k must be a positive integer")

    for section, key in (("models", "embedding"), ("models", "gemini")):
        if not isinstance(config[section][key], str) or not config[section][key].strip():
            raise ValueError(f"{section}.{key} must be a non-empty string")
