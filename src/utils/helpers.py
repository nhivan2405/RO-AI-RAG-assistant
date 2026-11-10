"""Thiết lập logging dùng chung mà không ghi dữ liệu nhạy cảm."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping


APP_LOGGER_NAME = "ro_ai_rag"
DEFAULT_LOGGING_CONFIG = {"level": "INFO", "file": "logs/app.log"}


def setup_logging(config: Mapping[str, Any] | None = None) -> logging.Logger:
    """Thiết lập file logger một lần từ cấu hình không nhạy cảm."""
    logging_config = _get_logging_config(config)
    log_path = Path(logging_config["file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(logging_config["level"].upper())
    logger.propagate = False
    _configure_file_handler(logger, log_path)
    return logger


def get_logger(component: str) -> logging.Logger:
    """Trả về logger con để dùng chung handler của ứng dụng."""
    return logging.getLogger(f"{APP_LOGGER_NAME}.{component}")


def _get_logging_config(config: Mapping[str, Any] | None) -> Mapping[str, str]:
    if config is None:
        from src.config import load_config

        return load_config()["logging"]
    return config.get("logging", DEFAULT_LOGGING_CONFIG)


def _configure_file_handler(logger: logging.Logger, log_path: Path) -> None:
    resolved_path = log_path.resolve()
    # Tái sử dụng handler cùng đường dẫn để tránh ghi log lặp mỗi lần khởi tạo.
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == resolved_path:
            return

    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(resolved_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
