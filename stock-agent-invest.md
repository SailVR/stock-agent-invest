# Personal Investment Research Agent

## 项目简介

Personal Investment Research Agent 是一个基于 Multi-Agent、RAG 和大语言模型的个人投研分析系统。

系统通过每日自动采集市场行情、新闻、公告和研报信息，结合用户持仓和自选股，生成市场分析、风险预警和投资策略建议。

核心目标：

* 自动化投研分析
* 持仓风险监控
* 热点板块发现
* 个股研究
* 每日投资日报生成

---

# 系统目标

用户每日上传：

* 持仓文件（CSV / Excel）
* 自选股列表

系统自动完成：

1. 市场复盘
2. 热点板块分析
3. 机构研报分析
4. 公司公告分析
5. 风险预警
6. 持仓跟踪
7. 调仓建议
8. 每日投资日报生成

---

# 总体架构

```text
                   ┌────────────────────┐
                   │ 持仓文件 / 自选股   │
                   └─────────┬──────────┘
                             │
                             ▼

                ┌────────────────────────┐
                │ Portfolio Agent        │
                └─────────┬──────────────┘
                          │

         ┌────────────────┼────────────────┐
         ▼                ▼                ▼

┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ Market Agent   │ │ Company Agent  │ │ Risk Agent     │
└───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        │                  │                  │
        ▼                  ▼                  ▼

                 ┌────────────────┐
                 │ Strategy Agent │
                 └───────┬────────┘
                         │
                         ▼
                 ┌────────────────┐
                 │ Report Agent   │
                 └────────────────┘
```

---

# 技术架构

## Frontend

* Streamlit

## Backend

* FastAPI

## Agent Framework

* LangGraph

## LLM

* GPT-5.5
* DeepSeek

## Embedding Model

* BAAI/bge-m3

## Vector Database

* FAISS

## Metadata Storage

* SQLite（MVP）
* DuckDB（升级版）

## Data Source

* AKShare

## Cache

* Redis

## Scheduler

* APScheduler

---

# 数据层设计

## 市场数据

通过 AKShare 获取：

* 上证指数
* 深证指数
* 创业板指数
* 北向资金
* 龙虎榜
* 涨跌停数据
* 板块涨跌幅

---

## 新闻数据

来源：

* 东方财富
* 财联社
* 证券时报
* 上海证券报

---

## 公告数据

包括：

* 业绩预告
* 回购公告
* 股东减持
* 问询函
* 立案调查

---

## 研报数据

包括：

* 机构评级
* 目标价
* 行业观点
* 推荐标的

---

# Event Layer

系统统一将所有文本信息转换为结构化事件。

示例：

原始新闻：

宁德时代发布股份回购公告。

结构化事件：

```json
{
  "event_type":"share_buyback",
  "company":"宁德时代",
  "sector":"新能源",
  "sentiment":"positive",
  "time":"2026-06-15"
}
```

---

# RAG架构

核心原则：

不存储原始新闻。

存储结构化事件。

流程：

```text
新闻
公告
研报
    ↓
Event Parser
    ↓
Event JSON
    ↓
Embedding
    ↓
FAISS
```

---

# FAISS设计

## 向量存储内容

* 新闻摘要
* 公告摘要
* 研报摘要
* Event Summary

---

## Metadata数据库

表结构：

```sql
docs

doc_id
company
sector
event_type
time
source
content
```

---

## 检索流程

```text
Query
   ↓
Embedding
   ↓
FAISS Search
   ↓
Doc ID
   ↓
SQLite Metadata
   ↓
时间过滤
持仓过滤
风险过滤
   ↓
LLM分析
```

---

# Tool Layer设计

Agent 不直接访问 AKShare。

统一通过 Tool Layer 调用。

结构：

```text
Tool Registry
│
├── Market Tools
├── Company Tools
├── News Tools
├── RAG Tools
├── Risk Tools
└── Portfolio Tools
```

---

## Market Tools

* get_market_index
* get_hot_sector
* get_capital_flow
* get_limit_up_stocks

---

## Company Tools

* get_company_news
* get_company_reports
* get_company_notice

---

## RAG Tools

* rag_search
* event_search

---

## Risk Tools

* risk_check
* announcement_check

---

# Agent设计

## Portfolio Agent

职责：

* 解析持仓文件
* 管理自选股
* 计算行业暴露
* 统计仓位分布

输出：

```json
{
  "holdings": [],
  "watchlist": []
}
```

---

## Market Agent

职责：

* 分析市场情绪
* 分析热点板块
* 统计资金流向
* 分析涨跌停结构

输出：

```json
{
  "market_sentiment":"strong",
  "hot_sectors":[]
}
```

---

## Company Agent

职责：

* 个股新闻分析
* 公告分析
* 研报分析
* 财务分析

输出：

```json
{
  "company":"宁德时代",
  "summary":"..."
}
```

---

## Risk Agent

职责：

* ST监控
* 问询函监控
* 减持监控
* 立案调查监控
* 财务风险监控

输出：

```json
{
  "risk_level":"high",
  "signals":[]
}
```

---

## Strategy Agent

职责：

综合分析：

* 市场状态
* 公司状态
* 风险状态
* 持仓状态

输出：

* 市场判断
* 持仓建议
* 风险提示
* 明日关注方向

---

## Report Agent

职责：

生成最终日报。

日报结构：

```text
# 今日市场

# 热点板块

# 持仓点评

# 风险预警

# 明日关注
```

---

# LangGraph工作流

```text
START

  │

  ▼

Portfolio Agent

  │

  ▼

┌───────────────────────┐
│      Parallel         │
├───────────────────────┤
│ Market Agent          │
│ Company Agent         │
│ Risk Agent            │
└───────────────────────┘

  │

  ▼

Strategy Agent

  │

  ▼

Report Agent

  │

  ▼

END
```

---

# 每日执行流程

18:00

* 获取市场数据
* 获取新闻
* 获取公告
* 获取研报

18:10

* Event抽取
* 更新FAISS索引

18:20

* 执行六大Agent

18:30

* 生成日报

---

# V1功能范围

* AKShare数据采集
* Event抽取
* FAISS RAG
* Tool Layer
* 六Agent协作
* 每日投资日报

---

# V2规划

新增：

* Graph RAG
* 热点轮动分析
* 龙头股识别
* 行业关系图谱

---

# V3规划

新增：

* 自动回测
* 调仓模拟
* 风险评分模型
* 个性化投资风格分析

---

# 最终目标

构建一个持续运行的个人投研智能体系统：

数据采集
→ Event抽取
→ FAISS知识库
→ Multi-Agent分析
→ 策略推理
→ 每日投资日报


```

```
