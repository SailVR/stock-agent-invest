"""
FAISS 向量索引 — 存储和检索向量。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from src.logs import get_logger

logger = get_logger(__name__)


class VectorStore:
    """FAISS 向量索引封装。

    使用 Inner Product (IP) + IDMap，搭配 L2 归一化 = cosine similarity。
    """

    def __init__(self, dim: int, index_path: str | Path | None = None):
        self.dim = dim
        self.index_path = str(index_path) if index_path else None

        if index_path and Path(index_path).exists():
            logger.info("[RAG] 加载 FAISS 索引 %s", index_path)
            self.index = faiss.read_index(str(index_path))
        else:
            base = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIDMap(base)
            logger.info("[RAG] 创建新 FAISS 索引，维度 %d", dim)

    def add(self, embeddings: np.ndarray, doc_ids: Sequence[int]) -> None:
        """添加向量 + 对应的文档 ID。

        Args:
            embeddings: shape=(n, dim) float32, 已归一化。
            doc_ids: 长度 n 的 int 序列。
        """
        if len(embeddings) != len(doc_ids):
            raise ValueError(f"embeddings({len(embeddings)}) 与 doc_ids({len(doc_ids)}) 长度不匹配")
        ids = np.array(doc_ids, dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)
        logger.info("[RAG] FAISS 添加 %d 条向量，总数 %d", len(embeddings), self.index.ntotal)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """搜索最相似的向量。

        Args:
            query_vec: shape=(dim,) 或 (1, dim) float32, 已归一化。
            top_k: 返回 top-K 结果。

        Returns:
            [(doc_id, score), ...] 按 score 降序。
        """
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, ids = self.index.search(query_vec, top_k)
        results = []
        for score, doc_id in zip(scores[0], ids[0]):
            if doc_id >= 0:  # FAISS 返回 -1 表示未找到
                results.append((int(doc_id), float(score)))
        return results

    def save(self, path: str | Path | None = None) -> None:
        """保存索引到磁盘。"""
        path = path or self.index_path
        if not path:
            raise ValueError("未指定保存路径")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        logger.info("[RAG] FAISS 索引已保存 %s (%d 条)", path, self.index.ntotal)

    @property
    def size(self) -> int:
        return self.index.ntotal
