"""Lazy RAG manager provider.

The embedding model is expensive to load, so the Flask app should not create a
RAGManager just to render the home page.
"""

from __future__ import annotations

import threading
from pathlib import Path

from src.logs import get_logger

logger = get_logger(__name__)


class LazyRAGProvider:
    """Create RAGManager only when RAG is actually used."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self._rag = None
        self._lock = threading.Lock()

    def get(self):
        if self._rag is None:
            with self._lock:
                if self._rag is None:
                    self._rag = self._create()
        return self._rag

    def _create(self):
        # Preserve the old behavior, but defer the cost until first RAG use.
        for rel_path in ["data/rag_index.faiss", "data/rag_docs.db"]:
            path = self.project_root / rel_path
            if path.exists():
                path.unlink()
                logger.info("[RAG] 清除旧索引 %s", rel_path)

        from src.rag import RAGManager

        logger.info("[RAG] 首次使用，开始初始化 RAGManager")
        return RAGManager()
