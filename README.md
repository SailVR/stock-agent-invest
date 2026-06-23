# 📊 Portfolio Agent — 智能投研助手

基于 **Flask + LangGraph + Multi-Agent + RAG** 的 A 股个人投研分析系统。系统支持上传持仓 CSV 或直接聊天提问，自动采集市场行情、财务数据、新闻信息，通过多个 Agent 协作完成持仓分析、风险评估、策略建议、历史检索和日报生成。

## 架构

```
用户持仓 CSV / 聊天提问
        │
Flask Web / SSE 流式响应
        │
Portfolio Analysis Service（上传分析流程编排）
        │
├──→ Market Agent    ──┐  (三路并行 ThreadPoolExecutor)
├──→ Company Agent   ──┤
├──→ Risk Agent      ──┘
        │
  merge_analysis（屏障节点，全部完成才放行）
        │
├── Strategy Agent  → LLM 综合策略建议
├── Report Agent    → LLM 生成投资日报
        │
        ▼
  Memory（SQLite 持久化）
   + RAG（懒加载，FAISS + BM25 混合检索 + JSONL 持久化）
```

## 功能

| Agent | 功能 | 数据源 |
|---|---|---|
| **Market Agent** | 大盘指数、涨停统计、LLM 情绪分析 | Tushare index_daily, AKShare zt_pool |
| **Company Agent** | ROE/负债率/毛利率/净利同比、新闻关键词检测 | Tushare fina_indicator, AKShare news |
| **Risk Agent** | ST/*ST 标记、停牌检测（过滤盘中暂停） | AKShare, Tushare suspend_d |
| **Strategy Agent** | 市场判断、持仓建议、风险提示、明日关注 | 综合 LLM |
| **Report Agent** | 生成投资日报 | 综合 LLM |
| **Memory** | 持仓快照、分析记录、用户偏好提取（LLM） | SQLite + LLM |
| **RAG** | 混合检索历史分析（向量语义 + BM25 关键词），首次使用才加载 | FAISS + BGE-Small-zh |

## 页面能力

| 页面 | 路径 | 说明 |
|---|---|---|
| 聊天与上传页 | `/` | 上传持仓 CSV，或直接输入股票/策略问题 |
| 流式分析接口 | `/analyze` | SSE 推送解析、市场、个股、风险、策略、日报生成进度 |
| 分析报告页 | `/result/<result_id>` | Dashboard 风格报告页，展示风险分布、市场概览、策略建议、投资日报和持仓明细 |
| 记忆接口 | `/api/memory/*` | 查看/清空用户偏好、历史持仓和最近分析记录 |

## 风险评级（5 级）

| 等级 | 颜色 | 含义 | LLM 判断参考 |
|---|---|---|---|
| **AAA** | 🟢 深绿 | 优秀 | ROE>15%、负债率<50%、毛利率>30%、利润增长、无负面 |
| **AA** | 🟢 浅绿 | 良好 | 大部分指标正常，1 个轻微瑕疵 |
| **A** | 🟡 黄 | 关注 | 1-2 个风险点，需跟踪 |
| **BB** | 🟠 橙 | 预警 | 多个风险信号，建议减仓 |
| **CC** | 🔴 红 | 危险 | ST/立案/ROE为负/严重亏损 |

## 分析过程展示

上传 CSV 分析时，网页会实时逐条打印分析过程：

```
📈 大盘: 上证指数 +1.2% | 深证成指 +2.5% | 涨停 145 只
📊 601288 农业银行: ROE=2.3% 负债率=93.5% 评级=a
📊 002463 沪电股份: ROE=7.8% 负债率=48.6% 评级=aa
📊 002826 易明医药: ROE=2.4% 负债率=24.0% 评级=a
🔍 风险排查完成
🔄 综合评级中...
🧠 生成策略建议...
📝 生成投资日报...
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cat > .env <<'EOF'
DASHSCOPE_API_KEY=你的千问APIKey
DEEPSEEK_API_KEY=你的DeepSeek APIKey
TUSHARE_API=你的Tushare Token
EOF

# 3. 启动
python app.py

# 4. 打开浏览器访问
open http://localhost:5001
```

## 环境变量

| 变量 | 说明 |
|---|---|
| `DASHSCOPE_API_KEY` | 千问 LLM API Key |
| `DEEPSEEK_API_KEY` | DeepSeek LLM API Key |
| `TUSHARE_API` | Tushare Pro API Key |

当前在 `src/config.py` 中通过 `LLM_PROVIDER` 切换 LLM：

```python
LLM_PROVIDER = "deepseek"   # 使用 DeepSeek V4 Flash
LLM_PROVIDER = "dashscope"  # 使用千问 qwen-turbo
```

默认值是 `dashscope`。如果要切换到 DeepSeek，需要修改 `src/config.py` 或进一步改造成从 `.env` 读取。

## 项目结构

```
stock-agent-invest/
│
├── app.py                          # Flask 入口：路由、Memory/RAG Provider 初始化、聊天流
├── requirements.txt                # Python 依赖
├── .env                            # API Key 配置（不提交 git）
│
├── src/                            # 核心逻辑
│   ├── config.py                   #   模型配置：LLM模型名/温度、Embedding模型、HF镜像地址
│   ├── utils.py                    #   工具函数：tushare_api_key()、dashscope_api_key()、safe_float()
│   ├── llm.py                      #   LLM 封装：chat() 对话、chat_json() JSON 模式（兼容DeepSeek）
│   ├── logs.py                     #   日志：控制台输出 + 滚动文件 (10MB × 5)
│   │
│   ├── agents/                     #   [Agent Layer] LangGraph Agent 定义与主图编排
│   │   ├── __init__.py             #      导出 Agent / State / 共享节点
│   │   └── portfolio_agents.py     #      Market/Company/Risk/Strategy/Report Agent + 合并逻辑
│   │
│   ├── services/                   #   [Service Layer] Web 路由背后的业务流程
│   │   ├── analysis_stream.py      #      上传 CSV 后的 SSE 流式分析流程
│   │   ├── rag_provider.py         #      RAGManager 懒加载，首次检索/写入才初始化
│   │   └── stock_lookup.py         #      股票名称 → 代码缓存与模糊匹配
│   │
│   ├── tools/                      #   [Tool Layer] 数据采集，Agent 不直接访问 Tushare/AKShare
│   │   ├── market_tools.py         #      get_market_index()、get_north_flow()、get_limit_up_stocks() → Tushare
│   │   ├── company_tools.py        #      get_financial_indicators()、fetch_stock_news() → Tushare + AKShare
│   │   └── stock_tools.py          #      股票名称映射、ST 检测、停牌检测、代码格式转换
│   │
│   └── rag/                        #   [RAG] 混合检索
│       ├── __init__.py             #      设置 HF_ENDPOINT = hf-mirror.com
│       ├── embeddings.py           #      BAAI/bge-small-zh-v1.5 嵌入，运行时自动探测维度
│       ├── vector_store.py         #      FAISS IndexFlatIP + IDMap
│       ├── documents.py            #      SQLite 文档元数据
│       └── rag_manager.py          #      统一接口：向量+BM25混合检索、JSONL持久化
│
├── memory/                         # [Memory] 持久化记忆
│   ├── schema.py                   #   建表：users、portfolios、stock_notes、user_interests
│   ├── long_term_memory.py         #   CRUD：持仓快照/分析记录/偏好读写
│   └── memory_extractor.py         #   LLM 从对话中提取用户关注偏好
│
├── templates/                      # [前端] Flask Jinja2 模板
│   ├── index.html                  #   聊天界面 + 左侧边栏（上传/偏好/历史）
│   └── result.html                 #   Dashboard 报告页（风险分布 + 市场概览 + 策略 + 日报 + 明细表）
│
├── tests/                          # [测试]
│   ├── test_market_tools.py        #   市场数据工具验证
│   ├── test_company_tools.py       #   公司数据工具验证
│   ├── test_llm.py                 #   大模型连通性测试（基础对话/股票分析/JSON输出）
│   ├── test_rag.py                 #   RAG 模块测试（Embedding/FAISS/SQLite/端到端检索）
│   └── test_graph.py               #   打印/保存所有 Agent 的 LangGraph 结构图
│
├── data/                           # [运行时数据]
│   ├── memory.db                   #   SQLite 记忆库（持仓快照/分析记录）
│   ├── rag_index.faiss             #   FAISS 向量索引（首次使用 RAG 时清除+重建）
│   ├── rag_docs.db                 #   SQLite 文档元数据（首次使用 RAG 时清除+重建）
│   └── rag_documents.jsonl         #   RAG 数据持久化（不删除，首次使用 RAG 时恢复）
│
├── generate_arch.py                # 架构图生成脚本
├── stock-agent-invest.md           # 原始设计文档
│
└── logs/                           # 运行日志（自动生成）
    └── agent.log                   #   滚动日志，每 10MB 轮转，保留 5 份
```

### 核心依赖

```
flask               → Web 服务 / SSE 流式响应
langgraph           → Agent 状态图 / 并行路由 / 屏障节点
tushare / akshare   → 数据源（财务 / 指数 / 涨停 / 新闻 / 停牌）
openai              → LLM 调用（兼容 DashScope 千问 + DeepSeek）
langchain-huggingface → Embedding 模型 BGE-Small-zh
faiss-cpu           → 向量检索
rank-bm25           → 关键词检索
```

## 混合检索（RAG）

RAG 使用 `LazyRAGProvider` 懒加载：打开首页和普通路由不会加载 embedding 模型；只有聊天命中历史检索，或上传分析完成后需要写入 RAG 时，才会初始化 `RAGManager`。

```
用户提问
    │
    ├──→ BGE-Small 转向量 → FAISS cosine similarity（语义）
    │
    └──→ BM25 关键词匹配 → 中文按字 / 英文按 token（关键词）
    │
    加权融合（默认各 0.5）
    │
    ├──→ 按综合得分排序
    └──→ 返回 top_k
```

- 首次使用 RAG 时清除 FAISS + SQLite，并从 `rag_documents.jsonl` 恢复
- Embedding 维度由模型实际输出自动探测，避免配置维度和 FAISS 索引维度不一致
- BM25 索引在首次搜索时自动重建

## 当前实现说明

- `src/agents/portfolio_agents.py` 内聚合了 5 个业务 Agent 和 1 个 Portfolio 主图；当前规模下便于查看整体流程，后续可继续拆成独立 Agent 文件。
- 上传分析为了实时向前端推送进度，实际由 `src/services/analysis_stream.py` 手动并行调用 Market / Company / Risk Agent。
- `portfolio_agent` 主图保留为标准 LangGraph 编排，已使用 fan-out + barrier 汇合 Market / Company / Risk 三路结果。
- 数据源访问统一收敛在 `src/tools/`，Agent 层只负责编排、Prompt 构造和结果合并。
- 结果页 `templates/result.html` 已改为 Dashboard 风格，并包含“仅供研究参考，不构成投资建议”的免责声明。

## 测试

```bash
# LLM 连通性测试
python tests/test_llm.py

# RAG 模块测试（首次需下载约 100MB 模型）
python tests/test_rag.py

# 查看所有 Agent 结构图
python tests/test_graph.py

# 工具函数测试
python tests/test_market_tools.py
python tests/test_company_tools.py
```

## 技术栈

| 组件 | 选型 |
|---|---|
| Web 框架 | Flask |
| Agent 框架 | LangGraph (v1) |
| 数据源 | Tushare Pro / AKShare |
| LLM | DeepSeek V4 Flash / 千问（可切换） |
| 文本嵌入 | BAAI/bge-small-zh-v1.5（维度自动探测） |
| 向量库 | FAISS (IndexFlatIP) |
| 关键词检索 | BM25 (rank-bm25) |
| 持久化 | SQLite + JSONL |
| 日志 | RotatingFileHandler (10MB × 5) |
| HF 镜像 | https://hf-mirror.com |
