"""RAG 模块 — 基于 FAISS + BGE-Small 的向量检索系统。

加载 RAG 模块时会自动设置 HF_ENDPOINT 镜像。
"""

from __future__ import annotations

import os

from src.config import HF_ENDPOINT

# 必须在导入 sentence_transformers 之前设置 HF 镜像
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

from .rag_manager import RAGManager

__all__ = ["RAGManager"]
