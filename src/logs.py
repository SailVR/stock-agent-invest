"""
Logging module — 统一日志配置，输出到控制台和日志文件。

用法：
    from src.logs import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "agent.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_BACKUP_COUNT = 5               # 保留最近 5 个文件

# 防止重复初始化
_initialized = False


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    max_bytes: int = _MAX_BYTES,
    backup_count: int = _BACKUP_COUNT,
) -> None:
    """配置全局日志：控制台 + 滚动文件。

    Args:
        level: 日志级别，默认 INFO。
        log_file: 日志文件路径，默认 logs/agent.log。
        max_bytes: 单个日志文件最大字节数。
        backup_count: 保留的备份文件数。
    """
    global _initialized
    if _initialized:
        return

    log_file = Path(log_file or _LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # ── 格式 ─────────────────────────────────────────────────────────
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_fmt = logging.Formatter(
        "%(levelname)-7s %(message)s",
    )

    # ── Handler: 滚动文件 ────────────────────────────────────────────
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_fmt)

    # ── Handler: 控制台 ──────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_fmt)

    # 清除已有 handler（防止重复添加）
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _initialized = True
    logging.getLogger(__name__).info(
        "日志已初始化 → %s (level=%s)", log_file, logging.getLevelName(level),
    )


def get_logger(name: str) -> logging.Logger:
    """获取一个 logger 实例。

    用法::

        from src.logs import get_logger
        logger = get_logger(__name__)

    首次调用时会自动初始化日志系统（如尚未初始化）。
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)
