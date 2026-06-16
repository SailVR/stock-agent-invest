# CLAUDE.md

给 Claude Code (claude.ai/code) 在本仓库中工作时的指引。

## 项目简介

基于 **LangGraph + Multi-Agent** 的 A 股个人投研分析系统。采集市场行情、财务数据、新闻信息，通过 6 个 Agent 协作完成持仓分析、风险评估、策略建议和日报生成。

**当前状态：** Agent 已全部实现，MVP 功能完整。所有 Agent 在 `app.py` 中定义为 LangGraph `StateGraph`。

## 运行测试

```bash
conda activate stock

# 全部测试
python tests/test_market_tools.py
python tests/test_company_tools.py
python tests/test_llm.py
python tests/test_rag.py
python tests/test_graph.py

# 启动服务
python app.py
```

## 架构

```
用户持仓 CSV / 聊天输入
    │
Portfolio Agent（主流程编排）
    │
    ├──→ Market Agent    ──┐  (三路并行, ThreadPoolExecutor)
    ├──→ Company Agent   ──┤
    └──→ Risk Agent      ──┘
            │
      merge_analysis（屏障节点，全部完成才放行）
            │
      Strategy Agent → Report Agent → Summary
```

**6 个 Agent：**

| Agent | 节点数 | 功能 | 数据源 |
|---|---|---|---|
| **Portfolio Agent** (主图) | 8 节点 | 编排所有子图 | - |
| **Market Agent** | 3 节点 | 指数行情、涨停池、LLM 情绪分析 | Tushare index_daily, AKShare zt_pool |
| **Company Agent** | 2 节点 | 财务指标 + 新闻 + LLM 评级(5级) | Tushare fina_indicator, AKShare news |
| **Risk Agent** | 3 节点 | ST 检测、停牌检测（过滤盘中暂停） | AKShare, Tushare suspend_d |
| **Strategy Agent** | 1 节点 | LLM 综合策略建议 | - |
| **Report Agent** | 1 节点 | LLM 生成投资日报 | - |

## Agent 工作流

```python
# 串行
parse_csv → enrich_holdings
    │
    ├──→ run_market_agent  (并行)
    ├──→ run_company_agent (并行)
    └──→ run_risk_agent    (并行)
            │
      merge_analysis（屏障路由：检查三个结果都就绪才放行）
            │
      run_strategy → run_report → generate_summary → END
```

## 关键设计模式

### 1. Tool Layer 规范

所有工具函数在 `src/tools/` 中，返回 **JSON 可序列化的 `dict` / `list[dict]`**，不返回原始 DataFrame。

| 模块 | 函数 |
|---|---|
| `market_tools.py` | `get_market_index()`, `get_north_flow()`, `get_limit_up_stocks()` |
| `company_tools.py` | `get_financial_indicators()`, `get_income_statement()`, `get_balance_sheet()`, `get_cash_flow()`, `fetch_stock_news()` |

### 2. LangGraph 屏障模式

Market/Company/Risk 三个 Agent 并行跑，`merge_analysis` 作为屏障节点检查三个结果都就绪才继续，通过 `merge_router` 条件路由实现：

```python
def merge_router(state) -> str:
    if state.get("market_assessments") and state.get("company_assessments") and state.get("risk_assessments") is not None:
        return "ready"
    return "wait"
```

### 3. RAG 混合检索（FAISS + BM25）

```
用户提问 → BGE-Small(384d) → FAISS cosine(语义)
         → BM25 关键词匹配(中文按字分词)
         → 加权融合(默认各0.5) → 返回 top_k
```

- 启动时清除 FAISS + SQLite，从 `rag_documents.jsonl` 恢复
- BM25 索引在首次搜索时自动重建

### 4. 风险评级 5 级

AAA(🟢)/AA(🟢)/A(🟡)/BB(🟠)/CC(🔴)，LLM 按 prompt 标准判断 + 规则兜底。

## 数据源说明

- **Tushare Pro** — 财务指标、指数行情、停牌数据，Token 从 `.env` 读取
- **AKShare** — 涨停池、新闻、ST 检测
- **LLM** — 千问或 DeepSeek，通过 `src/config.py` 的 `LLM_PROVIDER` 切换
- **HF 镜像** — `https://hf-mirror.com`（国内环境使用 BGE 模型）

## 启动流程

1. `app.py` 启动时清除旧 FAISS + SQLite → `RAGManager()` 从 `rag_documents.jsonl` 重建索引
2. Flask 监听 5000 端口，提供 SSE 流式接口
3. 上传 CSV → `/analyze` SSE 流 → 前端实时展示分析过程
4. 聊天输入 → `/chat` SSE 流 → LLM 流式回复

## 关键文件

| 文件 | 作用 |
|---|---|
| `app.py` | Flask 入口 + 6 个 Agent 定义 + SSE 流式响应 |
| `src/config.py` | 模型常量（LLM_MODEL, EMBEDDING_DIM, HF_ENDPOINT 等） |
| `src/llm.py` | LLM 调用封装，`chat()` 流式 / `chat_json()` JSON 容错解析 |
| `src/rag/rag_manager.py` | RAG 统一接口：add/search/save，FAISS+BM25+JSONL |
| `src/rag/embeddings.py` | BGE-Small-zh 嵌入，384维，langchain_huggingface |
| `memory/long_term_memory.py` | SQLite CRUD，持仓快照/分析记录/偏好 |
| `templates/index.html` | 聊天界面，SSE 流式展示分析过程 |
