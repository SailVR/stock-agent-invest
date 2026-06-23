"""
Embedding 模块 — 使用 BAAI/bge-small-zh-v1.5 通过 langchain-huggingface。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL, EMBEDDING_DEVICE
from src.logs import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """BGE-Small 嵌入模型封装。"""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        logger.info("[RAG] 加载 Embedding 模型 %s ...", model_name)
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        probe = self._model.embed_documents(["dimension probe"])
        self._dim = len(probe[0])
        logger.info("[RAG] Embedding 模型加载完成，维度 %d", self._dim)

    def encode(self, texts: str | Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        emb = self._model.embed_documents(list(texts))
        return np.array(emb, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim
