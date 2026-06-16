"""
RAG 管理器 — 统一 Embedding + FAISS + BM25 的混合检索接口。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.logs import get_logger
from src.rag.documents import DocumentStore
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
FAISS_INDEX_PATH = str(DATA_DIR / "rag_index.faiss")
DOC_DB_PATH = str(DATA_DIR / "rag_docs.db")
JSONL_PATH = str(DATA_DIR / "rag_documents.jsonl")


def _tokenize(text: str) -> list[str]:
    """中文 + 英文分词（按字/词拆分）。"""
    text = text.lower()
    # 英文数字按空格/标点分词
    tokens = re.findall(r"[a-z0-9.%@+-]+|[一-鿿]", text)
    return [t for t in tokens if len(t) > 0]


class RAGManager:
    """RAG 管理器 — 混合检索（向量 + BM25 关键词）。

    使用示例::

        rag = RAGManager()
        doc_id = rag.add_document("贵州茅台", "ROE=32% 负债率=18%...")
        results = rag.search("银行股风险")
    """

    def __init__(
        self,
        faiss_path: str = FAISS_INDEX_PATH,
        doc_db_path: str = DOC_DB_PATH,
        jsonl_path: str = JSONL_PATH,
    ):
        self.jsonl_path = jsonl_path
        self.embedder = EmbeddingModel()
        self.docs = DocumentStore(doc_db_path)
        self.vector_store = VectorStore(self.embedder.dim, faiss_path)

        # BM25 懒加载
        self._bm25: BM25Okapi | None = None
        self._bm25_doc_ids: list[int] = []
        self._bm25_tokenized: list[list[str]] = []

        # 从 JSONL 恢复索引
        self._load_from_jsonl()

    # ── JSONL 持久化 ────────────────────────────────────────────────

    def _append_to_jsonl(self, record: dict) -> None:
        """追加一条文档记录到 JSONL 文件。"""
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_from_jsonl(self) -> None:
        """启动时从 JSONL 恢复所有文档到 FAISS + SQLite。"""
        path = Path(self.jsonl_path)
        if not path.exists() or path.stat().st_size == 0:
            return
        count = 0
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # 直接写入 SQLite 和 FAISS，不再写回 JSONL
                    doc_id = self.docs.add(
                        company=record.get("company", ""),
                        content=record.get("content", ""),
                        sector=record.get("sector", ""),
                        event_type=record.get("event_type", "analysis"),
                        source=record.get("source", "agent"),
                        time=record.get("time"),
                    )
                    vec = self.embedder.encode(record.get("content", ""))
                    self.vector_store.add(vec.reshape(1, -1), [doc_id])
                    count += 1
                except Exception:
                    logger.warning("[RAG] JSONL 恢复失败，跳过一行")
        logger.info("[RAG] 从 JSONL 恢复 %d 篇文档", count)

    # ── BM25 索引重建 ────────────────────────────────────────────────

    def _rebuild_bm25(self) -> None:
        """从 SQLite 全量重建 BM25 索引。"""
        import sqlite3
        conn = sqlite3.connect(self.docs.db_path)
        rows = conn.execute("SELECT doc_id, content FROM documents").fetchall()
        conn.close()

        self._bm25_doc_ids = [r[0] for r in rows]
        self._bm25_tokenized = [_tokenize(r[1]) for r in rows]
        self._bm25 = BM25Okapi(self._bm25_tokenized)
        logger.debug("[RAG] BM25 索引重建完成 (%d 篇)", len(rows))

    def _ensure_bm25(self) -> None:
        if self._bm25 is None:
            self._rebuild_bm25()

    # ── 写入 ────────────────────────────────────────────────────────

    def add_document(
        self,
        company: str,
        content: str,
        sector: str = "",
        event_type: str = "analysis",
        source: str = "agent",
        time: str | None = None,
    ) -> int:
        """添加一条文档并建立向量索引 + BM25 索引。"""
        doc_id = self.docs.add(company, content, sector, event_type, source, time)
        # 向量索引
        vec = self.embedder.encode(content)
        self.vector_store.add(vec.reshape(1, -1), [doc_id])
        # BM25 标记为过期
        self._bm25 = None
        # 写入 JSONL 持久化
        self._append_to_jsonl({
            "company": company, "content": content, "sector": sector,
            "event_type": event_type, "source": source, "time": time,
        })
        logger.debug("[RAG] 添加文档 %d → %s (%s)", doc_id, company, event_type)
        return doc_id

    def add_analysis_result(self, analysis: dict[str, Any]) -> int | None:
        """将 Company Agent 的分析结果存入 RAG。"""
        code = analysis.get("code", "")
        name = analysis.get("name", "")
        summary = analysis.get("llm_summary", "")
        risk = analysis.get("risk_level", "")
        roe = analysis.get("fin_roe")
        debt = analysis.get("fin_debt_ratio")
        gm = analysis.get("fin_grossprofit_margin")
        npy = analysis.get("fin_net_profit_yoy")
        content = (
            f"【{code} {name}】"
            f"评级={risk}。{summary}"
            f"{' ROE='+str(roe)+'%' if roe else ''}"
            f"{' 负债率='+str(debt)+'%' if debt else ''}"
            f"{' 毛利率='+str(gm)+'%' if gm else ''}"
            f"{' 净利同比='+str(npy)+'%' if npy else ''}"
        )
        if content.strip("【】"):
            return self.add_document(
                company=f"{code} {name}".strip(),
                content=content,
                event_type="analysis",
                source="company_agent",
            )
        return None

    # ── 混合检索 ────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """混合检索：向量语义 + BM25 关键词，按加权分数融合。

        Args:
            query: 查询文本。
            top_k: 返回条数。
            vector_weight: 向量分数权重（默认 0.5）。
            bm25_weight: BM25 分数权重（默认 0.5）。

        Returns:
            [{doc_id, company, content, event_type, time, score}, ...]
        """
        # 1. 向量检索
        query_vec = self.embedder.encode(query)
        vec_hits = dict(self.vector_store.search(query_vec, top_k * 2))

        # 2. BM25 关键词检索
        self._ensure_bm25()
        tokenized_query = _tokenize(query)
        bm25_scores = self._bm25.get_scores(tokenized_query) if self._bm25 is not None else np.array([])

        # 3. 融合打分
        # 归一化 BM25 分数到 [0, 1]
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1
        all_scores: dict[int, float] = {}
        for i, doc_id in enumerate(self._bm25_doc_ids):
            # 向量分数（缺失则 0）
            vec_score = vec_hits.get(doc_id, 0.0)
            # BM25 分数（归一化）
            bm25_score = bm25_scores[i] / max_bm25 if max_bm25 > 0 else 0
            # 加权融合
            combined = vec_score * vector_weight + bm25_score * bm25_weight
            if combined > 0:
                all_scores[doc_id] = combined

        # 4. 排序取 top_k
        sorted_ids = sorted(all_scores, key=all_scores.get, reverse=True)[:top_k]
        results = []
        for doc_id in sorted_ids:
            meta = self.docs.get(doc_id)
            if meta:
                meta["score"] = round(all_scores[doc_id], 4)
                results.append(meta)
        return results

    # ── 持久化 ──────────────────────────────────────────────────────

    def save(self) -> None:
        """保存 FAISS 索引到磁盘。"""
        self.vector_store.save()
        logger.info("[RAG] 已保存 (%d 篇文档, BM25 待重建)", self.docs.count())

    @property
    def stats(self) -> dict:
        return {
            "documents": self.docs.count(),
            "vectors": self.vector_store.size,
        }
