# CLAUDE.md

给 Claude Code / Codex 在本仓库中工作时的项目指引。

## 项目简介

这是一个基于 **Flask + LangGraph + Multi-Agent + RAG** 的 A 股个人投研分析系统。系统支持上传持仓 CSV 或直接聊天提问，自动采集市场行情、财务数据、新闻与风险信息，通过多个 Agent 协作生成持仓风险评级、策略建议和投资日报。

当前代码已经从单文件 Demo 拆成分层结构：

- `app.py`：Flask 路由入口，负责页面、上传、聊天和 Memory API。
- `src/agents/`：LangGraph Agent 定义与 Portfolio 主图。
- `src/services/`：Web 路由背后的业务流程，如 SSE 分析流、RAG 懒加载。
- `src/tools/`：所有 Tushare / AKShare 数据源访问。
- `src/rag/`：FAISS + BM25 混合检索。
- `memory/`：SQLite 长期记忆。

## 运行与测试

```bash
conda activate stock
pip install -r requirements.txt

# 启动服务，默认端口 5001
python app.py

# 工具层测试
python tests/test_market_tools.py
python tests/test_company_tools.py

# LLM / RAG / Agent 图测试
python tests/test_llm.py
python tests/test_rag.py
python tests/test_graph.py
```

注意：测试依赖外部服务和模型下载，需要 `.env` 中有 `DASHSCOPE_API_KEY` / `DEEPSEEK_API_KEY` / `TUSHARE_API`，并安装完整依赖。

## 当前架构

```text
用户持仓 CSV / 聊天输入
        │
Flask Web / SSE
        │
src/services/analysis_stream.py
        │
├── Market Agent  ──┐
├── Company Agent ──┤  三路并行 ThreadPoolExecutor
└── Risk Agent    ──┘
        │
merge_analysis
        │
Strategy Agent
        │
Report Agent
        │
Memory + RAG
```

`portfolio_agent` 主图也保留在 `src/agents/portfolio_agents.py` 中，使用 LangGraph fan-out + barrier：

```python
parse_csv -> enrich_holdings
enrich_holdings -> run_market_agent
enrich_holdings -> run_company_agent
enrich_holdings -> run_risk_agent
["run_market_agent", "run_company_agent", "run_risk_agent"] -> merge_analysis
merge_analysis -> run_strategy_agent -> run_report_agent -> generate_summary
```

上传分析为了实时推送进度，实际走 `src/services/analysis_stream.py` 手动并行调用三个子 Agent。

## Agent 列表

| Agent           | 位置                               | 功能                                          |
| --------------- | ---------------------------------- | --------------------------------------------- |
| Portfolio Agent | `src/agents/portfolio_agents.py` | 主图编排 CSV 解析、三路分析、合并、策略、日报 |
| Market Agent    | `src/agents/portfolio_agents.py` | 大盘指数、涨停池、市场情绪分析                |
| Company Agent   | `src/agents/portfolio_agents.py` | 财务指标、新闻标题、LLM 基本面评级            |
| Risk Agent      | `src/agents/portfolio_agents.py` | ST / 退市 / 停牌等硬风险检测                  |
| Strategy Agent  | `src/agents/portfolio_agents.py` | 综合市场、个股、风险信号生成策略建议          |
| Report Agent    | `src/agents/portfolio_agents.py` | 生成投资日报                                  |

当前所有 Agent 暂集中在一个文件中，便于查看完整流程。若继续扩大，可以拆成 `states.py`、`market_agent.py`、`company_agent.py`、`risk_agent.py`、`strategy_agent.py`、`report_agent.py`、`portfolio_graph.py`。

## 分层约定

### Tool Layer

所有数据源访问必须走 `src/tools/`，Agent 层不要直接 import `akshare` / `tushare`。

| 模块                           | 职责                                           |
| ------------------------------ | ---------------------------------------------- |
| `src/tools/market_tools.py`  | 市场指数、北向资金、涨停池                     |
| `src/tools/company_tools.py` | 财务指标、利润表、资产负债表、现金流、个股新闻 |
| `src/tools/stock_tools.py`   | 股票名称映射、ST 检测、停牌检测、代码格式转换  |

工具函数应该返回 JSON 可序列化的 `dict` / `list[dict]`，不要把 DataFrame 泄露给 Agent。

### Service Layer

| 文件                                | 职责                                          |
| ----------------------------------- | --------------------------------------------- |
| `src/services/analysis_stream.py` | 上传 CSV 后的 SSE 流式分析流程                |
| `src/services/rag_provider.py`    | RAGManager 懒加载，避免首页启动加载 embedding |
| `src/services/stock_lookup.py`    | 股票名称到代码的缓存和模糊匹配                |

`app.py` 目前仍包含较长的 `chat_api()`，后续可拆到 `src/services/chat_stream.py`。

## RAG 说明

RAG 使用 `LazyRAGProvider` 懒加载：

- 打开首页不会加载 embedding 模型。
- 聊天命中历史检索或上传分析完成后写入 RAG 时，才初始化 `RAGManager`。
- 首次初始化时清除 `data/rag_index.faiss` 和 `data/rag_docs.db`，再从 `data/rag_documents.jsonl` 恢复。
- `EmbeddingModel` 会运行时自动探测真实向量维度，避免 FAISS 索引维度和模型输出不一致。

检索流程：

```text
query -> BGE-Small 向量 -> FAISS cosine
      -> BM25 关键词
      -> 加权融合 -> top_k
```

## Memory 说明

SQLite 记忆库在 `memory/` 下封装：

- `portfolios`：持仓快照
- `stock_notes`：个股分析记录
- `user_interests`：用户偏好

`memory/memory_extractor.py` 会调用 LLM 从聊天中提取偏好。当前 `src/llm.py::chat_json()` 对 JSON 数组支持仍可继续增强。

## 页面说明

| 页面       | 文件 / 路径                                  | 说明                                                       |
| ---------- | -------------------------------------------- | ---------------------------------------------------------- |
| 首页       | `templates/index.html` / `/`             | 聊天、上传持仓、历史/偏好侧栏                              |
| 分析流     | `/analyze`                                 | SSE 返回分析进度                                           |
| 报告页     | `templates/result.html` / `/result/<id>` | Dashboard 风格报告页：风险分布、市场概览、策略、日报、明细 |
| Memory API | `/api/memory/*`                            | 偏好、历史、清空接口                                       |

报告页包含免责声明：模型输出仅供研究参考，不构成投资建议。

## 配置

配置集中在 `src/config.py`。

当前 `LLM_PROVIDER` 写死为：

```python
LLM_PROVIDER = "dashscope"
```

可切换为 `"deepseek"`，但目前不是从 `.env` 读取。若后续优化，建议改成：

```python
LLM_PROVIDER = get_env("LLM_PROVIDER", "dashscope")
```

Embedding：

- 模型：`BAAI/bge-small-zh-v1.5`
- 设备：CPU
- HF 镜像：`https://hf-mirror.com`
- 维度：运行时自动探测

## 运行产物与 Git

`.gitignore` 已忽略：

- `.env`
- `__pycache__` / `*.pyc`
- `logs/*`
- `data/*`
- 虚拟环境、构建产物、编辑器目录

保留：

- `data/.gitkeep`
- `logs/.gitkeep`

当前项目不再使用 `uploads/`，该目录已删除。

## 重要注意事项

- 不要把 `.env`、数据库、FAISS 索引、日志提交到仓库。
- 不要在 Agent 层直接访问 Tushare / AKShare，统一走 `src/tools/`。
- 若改 RAG embedding 模型，确认 `EmbeddingModel.dim` 和 FAISS 索引维度仍由运行时探测。
- 若改上传分析流程，注意前端依赖 SSE 事件：`parsed`、`progress`、`complete`、`error`。
- 金融输出必须保持“仅供研究参考，不构成投资建议”的语义。
