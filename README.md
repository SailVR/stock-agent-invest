# 📊 Portfolio Agent — 智能投研助手

基于 **LangGraph + Multi-Agent** 的 A 股个人投研分析系统。自动采集市场行情、财务数据、新闻信息，通过 6 个 Agent 协作完成持仓分析、风险评估、策略建议和日报生成。

## 架构

```
用户持仓 CSV / 聊天提问
        │
Portfolio Agent（主流程编排）
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
   + RAG（FAISS + BM25 混合检索 + JSONL 持久化）
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
| **RAG** | 混合检索历史分析（向量语义 + BM25 关键词） | FAISS + BGE-Small-zh |

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
cp .env.example .env
# 编辑 .env 填入 API Key

# 3. 启动
python app.py

# 4. 打开浏览器访问
open http://localhost:5000
```

## 环境变量

| 变量 | 说明 |
|---|---|
| `DASHSCOPE_API_KEY` | 千问 LLM API Key |
| `DEEPSEEK_API_KEY` | DeepSeek LLM API Key |
| `TUSHARE_API` | Tushare Pro API Key |

在 `src/config.py` 中通过 `LLM_PROVIDER` 切换 LLM：

```python
LLM_PROVIDER = "deepseek"   # 使用 DeepSeek V4 Flash
LLM_PROVIDER = "dashscope"  # 使用千问 qwen-turbo
```

## 项目结构

```
stock-agent-invest/
│
├── app.py                          # 入口：Flask + LangGraph 6 个 Agent 定义
├── requirements.txt                # Python 依赖
├── .env                            # API Key 配置（不提交 git）
├── .env.example                    # API Key 模板
│
├── src/                            # 核心逻辑
│   ├── config.py                   #   模型配置：LLM模型名/温度、Embedding模型/维度、HF镜像地址
│   ├── utils.py                    #   工具函数：tushare_api_key()、dashscope_api_key()、safe_float()
│   ├── llm.py                      #   LLM 封装：chat() 流式对话、chat_json() JSON 模式（兼容DeepSeek）
│   ├── logs.py                     #   日志：控制台输出 + 滚动文件 (10MB × 5)
│   │
│   ├── tools/                      #   [Tool Layer] 数据采集
│   │   ├── market_tools.py         #      get_market_index()、get_north_flow()、get_limit_up_stocks() → Tushare
│   │   └── company_tools.py        #      get_financial_indicators()、fetch_stock_news() → Tushare + AKShare
│   │
│   └── rag/                        #   [RAG] 混合检索
│       ├── __init__.py             #      设置 HF_ENDPOINT = hf-mirror.com
│       ├── embeddings.py           #      BAAI/bge-small-zh-v1.5 嵌入 → 384维向量
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
│   └── result.html                 #   分析结果页（评级表格 + 策略 + 日报）
│
├── tests/                          # [测试]
│   ├── test_market_tools.py        #   市场数据工具验证
│   ├── test_company_tools.py       #   公司数据工具验证
│   ├── test_llm.py                 #   大模型连通性测试（基础对话/股票分析/JSON输出）
│   ├── test_rag.py                 #   RAG 模块测试（Embedding/FAISS/SQLite/端到端检索）
│   └── test_graph.py               #   打印/保存所有 Agent 的 LangGraph 结构图
│
├── data/                           # [运行时数据] 启动自动重建
│   ├── memory.db                   #   SQLite 记忆库（持仓快照/分析记录）
│   ├── rag_index.faiss             #   FAISS 向量索引（启动清除+重建）
│   ├── rag_docs.db                 #   SQLite 文档元数据（启动清除+重建）
│   └── rag_documents.jsonl         #   RAG 数据持久化（不删除，启动时恢复）
│
├── ARCHITECTURE.md                 # Mermaid 架构图
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

- 启动时自动清除 FAISS + SQLite，从 `rag_documents.jsonl` 恢复
- BM25 索引在首次搜索时自动重建

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
| 文本嵌入 | BAAI/bge-small-zh-v1.5 (384d) |
| 向量库 | FAISS (IndexFlatIP) |
| 关键词检索 | BM25 (rank-bm25) |
| 持久化 | SQLite + JSONL |
| 日志 | RotatingFileHandler (10MB × 5) |
| HF 镜像 | https://hf-mirror.com |
