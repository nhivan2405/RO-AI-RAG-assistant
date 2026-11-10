"""Đọc và kiểm tra cấu hình không nhạy cảm của ứng dụng."""
from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "chunking": {
        "strategy": "recursive",
        "chunk_size": 700,
        "chunk_overlap": 100,
    },
    "retrieval": {"top_k": 5, "min_score": 0.5},
    "models": {
        "embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "gemini": "gemini-2.5-flash",
    },
    "vectordb": {
        "storage_path": "qdrant_storage",
        "collection_name": "rag_demo",
    },
    "logging": {"level": "INFO", "file": "logs/app.log"},
    "prompts": {"context_token_budget": 1200},
    "llm": {"timeout_seconds": 30},
    "evaluation": {
        "embedding_models": [
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
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
    if strategy not in {"fixed", "recursive", "structure_aware"}:
        raise ValueError(
            "chunking.strategy must be 'fixed', 'recursive', or 'structure_aware'"
        )

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

    min_score = config["retrieval"]["min_score"]
    if (
        not isinstance(min_score, (int, float))
        or isinstance(min_score, bool)
        or not 0 <= min_score <= 1
    ):
        raise ValueError("retrieval.min_score must be a number between 0 and 1")

    for section, key in (("models", "embedding"), ("models", "gemini")):
        if not isinstance(config[section][key], str) or not config[section][key].strip():
            raise ValueError(f"{section}.{key} must be a non-empty string")

    for key in ("storage_path", "collection_name"):
        value = config["vectordb"][key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"vectordb.{key} must be a non-empty string")

    logging_config = config["logging"]
    level = logging_config["level"]
    if not isinstance(level, str) or logging.getLevelName(level.upper()) == f"Level {level.upper()}":
        raise ValueError("logging.level must be a valid logging level")
    log_file = logging_config["file"]
    if not isinstance(log_file, str) or not log_file.strip():
        raise ValueError("logging.file must be a non-empty string")

    context_token_budget = config["prompts"]["context_token_budget"]
    if (
        not isinstance(context_token_budget, int)
        or isinstance(context_token_budget, bool)
        or context_token_budget <= 0
    ):
        raise ValueError("prompts.context_token_budget must be a positive integer")

    timeout_seconds = config["llm"]["timeout_seconds"]
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ValueError("llm.timeout_seconds must be a positive integer")

    evaluation_models = config["evaluation"]["embedding_models"]
    if not isinstance(evaluation_models, list) or not evaluation_models:
        raise ValueError("evaluation.embedding_models must be a non-empty list")
    if not all(isinstance(model, str) and model.strip() for model in evaluation_models):
        raise ValueError("evaluation.embedding_models must contain non-empty strings")
