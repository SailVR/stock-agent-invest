#!/usr/bin/env python3
"""验证 RAG 模块：Embedding 模型加载、向量维度、语义检索。

用法：
    python tests/test_rag.py          # 测试 embedding + 向量检索
    python tests/test_rag.py --full   # 全部测试（含文档写入）
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import HF_ENDPOINT
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

from src.logs import setup_logging
setup_logging()

import numpy as np
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.documents import DocumentStore


def _sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_embedding_model() -> None:
    """1. 测试 BGE-Small 模型加载和编码。"""
    _sep("1. EmbeddingModel — 模型加载 & 编码")

    t0 = time.time()
    model = EmbeddingModel()
    print(f"  模型加载耗时: {time.time()-t0:.1f}s")
    print(f"  向量维度: {model.dim}")

    texts = [
        "贵州茅台2025年ROE为32.5%，毛利率91.8%",
        "农业银行不良贷款率上升，资产质量承压",
        "新能源板块今日资金净流入50亿元",
    ]
    vecs = model.encode(texts)
    print(f"  输入 {len(texts)} 条文本")
    print(f"  输出向量 shape: {vecs.shape}")
    assert vecs.shape == (3, model.dim), f"维度错误: {vecs.shape}"
    assert vecs.dtype == np.float32, f"类型错误: {vecs.dtype}"
    # 检查归一化（各行 L2 范数 ≈ 1）
    norms = np.linalg.norm(vecs, axis=1)
    print(f"  L2 范数: {[round(n, 4) for n in norms]}")
    assert all(abs(n - 1.0) < 0.01 for n in norms), "向量未归一化"
    print("  ✅ Embedding 模型测试通过\n")


def test_vector_store() -> None:
    """2. 测试 FAISS 检索。"""
    _sep("2. VectorStore — 向量存储 & 检索")

    dim = 32
    store = VectorStore(dim)

    # 写入 5 条随机向量
    rng = np.random.RandomState(42)
    vecs = rng.randn(5, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # 归一化
    store.add(vecs, [101, 102, 103, 104, 105])
    print(f"  写入后向量数: {store.size}")
    assert store.size == 5

    # 检索（用第一条向量查）
    results = store.search(vecs[0], top_k=3)
    print(f"  检索结果: {results}")
    assert len(results) >= 1
    assert results[0][0] == 101, f"最相似应为 101，实际 {results[0]}"
    print("  ✅ VectorStore 测试通过\n")


def test_document_store() -> None:
    """3. 测试 SQLite 文档存储。"""
    _sep("3. DocumentStore — 元数据存储 & 查询")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = DocumentStore(db_path)
        id1 = store.add("贵州茅台", "ROE=32.5%，毛利率91.8%，建议关注", event_type="analysis")
        id2 = store.add("农业银行", "不良贷款率上升，风险偏高", event_type="analysis")
        id3 = store.add("市场", "今日市场情绪偏强，涨停145只", event_type="market")
        print(f"  写入 3 条文档，ID: {id1}, {id2}, {id3}")
        assert id1 > 0

        # 按公司查询
        rows = store.search_by_company("茅台")
        print(f"  查询 '茅台' 找到 {len(rows)} 条")
        assert len(rows) >= 1

        # 按类型查询
        rows = store.search_by_event("market")
        print(f"  查询 event=market 找到 {len(rows)} 条")
        assert len(rows) >= 1

        print(f"  总文档数: {store.count()}")
        assert store.count() == 3
        print("  ✅ DocumentStore 测试通过\n")
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_semantic_search() -> None:
    """4. 端到端测试：写入 → 语义检索。"""
    _sep("4. 端到端 — 写入 + 语义检索")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    index_path = db_path + ".faiss"

    try:
        from src.rag.rag_manager import RAGManager
        rag = RAGManager(faiss_path=index_path, doc_db_path=db_path)

        # 写入几条分析
        docs = [
            ("600519 贵州茅台", "贵州茅台ROE=32.5%，毛利率91.8%，净利同比+15%，基本面优秀", "analysis"),
            ("601288 农业银行", "农业银行ROE=2.3%，负债率93.5%，净利同比+4.5%，估值偏低", "analysis"),
            ("002463 沪电股份", "沪电股份毛利率35.6%，净利同比-8.3%，营收承压", "analysis"),
            ("市场情绪", "今日上证指数+1.2%，涨停145只，市场情绪偏强", "market"),
        ]
        for company, content, event_type in docs:
            rag.add_document(company, content, event_type=event_type)
        rag.save()
        print(f"  写入 {len(docs)} 条文档")

        # 语义检索测试
        queries = [
            ("贵州茅台财务状况", "贵州茅台"),
            ("银行股票风险", "农业银行"),
            ("今天市场怎么样", "市场"),
        ]
        for query, expected in queries:
            results = rag.search(query, top_k=2)
            top_company = results[0]["company"] if results else "无"
            print(f"  查询: '{query}' → 最相关: {top_company}")
            assert expected in top_company, f"预期 {expected}，实际 {top_company}"

        print("  ✅ 语义检索测试通过\n")
    finally:
        Path(db_path).unlink(missing_ok=True)
        Path(index_path).unlink(missing_ok=True)


def main() -> None:
    print(">>> 开始验证 RAG 模块 <<<")

    full = "--full" in sys.argv
    tests = [
        ("Embedding 模型", test_embedding_model),
        ("FAISS 向量检索", test_vector_store),
        ("文档元数据存储", test_document_store),
    ]
    if full:
        tests.append(("端到端语义检索", test_semantic_search))

    errors = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n  ❌ {name} 失败: {e}")
            errors.append(name)

    print(f"\n{'=' * 60}")
    if errors:
        print(f"  部分失败: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("  全部验证通过 ✅")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
