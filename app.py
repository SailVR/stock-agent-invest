#!/usr/bin/env python3
"""
Flask + LangGraph demo — Portfolio Agent (单 Agent).

Upload a CSV 持仓文件 → agent 解析 → 展示持仓概览。
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
import uuid
from typing import Any, TypedDict

import akshare as ak
import pandas as pd
import tushare as ts
from flask import Flask, render_template, request, Response
from langgraph.graph import StateGraph, END

from memory.long_term_memory import LongTermMemory, DEFAULT_USER
from src.rag import RAGManager
from memory.memory_extractor import MemoryExtractor
from src.llm import chat
from src.utils import tushare_api_key
from src.logs import get_logger, setup_logging

logger = get_logger(__name__)

# ── Tushare Pro ─────────────────────────────────────────────────────────
_ts_pro = ts.pro_api(
    tushare_api_key()
)

app = Flask(__name__)
app.secret_key = os.urandom(16).hex()

# ── 结果缓存（内存，分析完成后前端跳转来取） ────────────────────────
_result_cache: dict[str, dict] = {}
_RESULT_TTL = 300

# ── Memory ─────────────────────────────────────────────────────────────
memory = LongTermMemory()
extractor = MemoryExtractor()

# 清除旧 RAG 索引（每次启动重建，数据从 rag_documents.jsonl 恢复）
for _f in ["data/rag_index.faiss", "data/rag_docs.db"]:
    _p = os.path.join(os.path.dirname(__file__), _f)
    if os.path.exists(_p):
        os.remove(_p)
        logger.info("[RAG] 清除旧索引 %s", _f)
rag = RAGManager()


# ════════════════════════════════════════════════════════════════════
# 核心逻辑（与 LangGraph 共享）
# ════════════════════════════════════════════════════════════════════


class PortfolioState(TypedDict):
    csv_content: str
    holdings: list[dict[str, Any]]
    summary: dict[str, Any]
    analysis: list[dict[str, Any]]
    risk_signals: list[str]
    strategy_result: dict[str, Any]
    report_text: str
    error: str | None


def _build_stock_name_map() -> dict[str, str]:
    try:
        df = ak.stock_info_a_code_name()
        return dict(zip(df["code"], df["name"]))
    except Exception:
        logger.warning("Failed to fetch stock name map.")
        return {}


def parse_csv(state: PortfolioState) -> dict:
    csv_text = state["csv_content"]
    if not csv_text.strip():
        return {"error": "CSV 内容为空", "holdings": []}

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return {"error": "CSV 无法解析或没有数据行", "holdings": []}

    holdings = []
    for row in rows:
        cleaned = {k.strip().lower(): v.strip() for k, v in row.items()}
        code = cleaned.get("code") or cleaned.get("股票代码")
        if not code:
            continue
        holdings.append({
            "code": code,
            "date": cleaned.get("date", ""),
            "prey": _safe_float(cleaned.get("prey")),
        })

    if not holdings:
        return {"error": "未找到有效的股票代码列", "holdings": []}

    logger.info("Parsed %d holdings from CSV.", len(holdings))
    return {"holdings": holdings, "error": None}


def enrich_holdings(state: PortfolioState) -> dict:
    holdings = state["holdings"]
    name_map = _build_stock_name_map()
    enriched = [{**h, "name": name_map.get(h["code"], "未知")} for h in holdings]
    logger.info("Enriched %d holdings with names.", len(enriched))
    return {"holdings": enriched}


def generate_summary(state: PortfolioState) -> dict:
    holdings = state["holdings"]
    prey_vals = [h["prey"] for h in holdings if h["prey"] is not None]
    summary: dict[str, Any] = {"total_stocks": len(holdings)}
    if prey_vals:
        summary.update(
            prey_mean=round(sum(prey_vals) / len(prey_vals), 4),
            prey_max=max(prey_vals),
            prey_min=min(prey_vals),
        )
    return {"summary": summary}


# ════════════════════════════════════════════════════════════════════
# Market Agent — 市场情绪 + 涨跌停分析
# ════════════════════════════════════════════════════════════════════

class MarketState(TypedDict):
    holdings: list[dict[str, Any]]
    market_indices: dict[str, Any]
    limit_up_codes: list[str]
    market_assessments: list[dict[str, Any]]
    sentiment_analysis: str
    error: str | None


def _fetch_market_data(state: MarketState) -> dict:
    """获取大盘指数 & 今日涨停池。"""
    indices, limit_codes = {}, []
    for ts_c, name in [("000001.SH", "上证指数"), ("399001.SZ", "深证成指"),
                       ("399006.SZ", "创业板指")]:
        try:
            logger.info("[Tool] Tushare index_daily → %s (%s)", name, ts_c)
            df = _ts_pro.index_daily(ts_code=ts_c, start_date="", end_date="")
            if df is not None and not df.empty:
                r = df.iloc[0]
                indices[name] = {
                    "close": _safe_float(r.get("close")),
                    "pct_chg": _safe_float(r.get("pct_chg")),
                }
                logger.info("[Tool] index_daily 返回 %s = %.2f (%.2f%%)", name, indices[name]["close"], indices[name]["pct_chg"])
        except Exception:
            logger.warning("[Tool] index_daily 失败 %s", name)
    try:
        logger.info("[Tool] AKShare stock_zt_pool_em → 今日涨停池")
        limit_df = ak.stock_zt_pool_em(date=time.strftime("%Y%m%d"))
        time.sleep(0.3)
        if limit_df is not None and not limit_df.empty:
            limit_codes = limit_df["代码"].astype(str).tolist()
            logger.info("[Tool] 涨停池返回 %d 只涨停股", len(limit_codes))
    except Exception:
        pass
    return {"market_indices": indices, "limit_up_codes": limit_codes}


def _llm_assess_market(state: MarketState) -> dict:
    """LLM 分析市场情绪。"""
    indices = state.get("market_indices", {})
    summary = "; ".join(f"{k} 涨跌幅 {v.get('pct_chg', '?')}%" for k, v in indices.items())
    limit_cnt = len(state.get("limit_up_codes", []))

    prompt = (
        f"今日市场数据：{summary}；涨停 {limit_cnt} 只。\n"
        f"请分析当前市场情绪，给出判断（strong / neutral / weak）和简短理由。"
    )
    try:
        logger.info("[LLM] Market Agent 分析市场情绪")
        logger.info("[LLM] Prompt:\n%s", prompt)
        analysis = chat(
            "你是一个A股市场分析专家，请根据指数和涨停数据判断市场情绪。",
            prompt,
        )
        logger.info("[LLM] Market Agent 返回: %s", analysis[:200])
    except Exception as e:
        logger.warning("Market Agent LLM 调用失败: %s", e)
        analysis = f"市场数据：{summary}，涨停 {limit_cnt} 只"

    return {"sentiment_analysis": analysis}


market_assess_limit_up = lambda s: {
    "market_assessments": [
        {"code": h["code"], "name": h["name"],
         "is_limit_up": h["code"] in s.get("limit_up_codes", [])}
        for h in s.get("holdings", [])
    ]
}

market_wf = StateGraph(MarketState)
market_wf.add_node("fetch_market_data", _fetch_market_data)
market_wf.add_node("assess_limit_up", market_assess_limit_up)
market_wf.add_node("llm_assess", _llm_assess_market)
market_wf.set_entry_point("fetch_market_data")
market_wf.add_edge("fetch_market_data", "assess_limit_up")
market_wf.add_edge("assess_limit_up", "llm_assess")
market_wf.add_edge("llm_assess", END)
market_agent = market_wf.compile()


# ════════════════════════════════════════════════════════════════════
# Company Agent — 财务指标 + 新闻情绪
# ════════════════════════════════════════════════════════════════════

class CompanyState(TypedDict):
    holdings: list[dict[str, Any]]
    raw_data: list[dict[str, Any]]
    company_assessments: list[dict[str, Any]]
    error: str | None


def _code_to_tscode(code: str) -> str:
    code = code.strip().zfill(6)
    suffix = "SH" if code.startswith(("6", "9")) else "SZ"
    return f"{code}.{suffix}"


def _collect_company_data(state: CompanyState) -> dict:
    """逐股拉财务 + 新闻原始数据，不做判断。"""
    raw = []
    for h in state.get("holdings", []):
        code, name = h.get("code", ""), h.get("name", "")
        ts_code = _code_to_tscode(code)
        item: dict[str, Any] = {
            "code": code, "name": name, "ts_code": ts_code,
            "fin_roe": None, "fin_debt_ratio": None,
            "fin_grossprofit_margin": None, "fin_net_profit_yoy": None,
            "news_titles": [],
        }
        try:
            logger.info("[Tool] Tushare fina_indicator → %s", ts_code)
            df = _ts_pro.query(
                "fina_indicator", ts_code=ts_code,
                start_date="", end_date="",
                fields="eps,roe,grossprofit_margin,debt_to_assets,netprofit_yoy,or_yoy",
            )
            if df is not None and not df.empty:
                r = df.iloc[0]
                item.update(
                    fin_roe=_safe_pct(r.get("roe")),
                    fin_debt_ratio=_safe_pct(r.get("debt_to_assets")),
                    fin_grossprofit_margin=_safe_pct(r.get("grossprofit_margin")),
                    fin_net_profit_yoy=_safe_pct(r.get("netprofit_yoy"), 10000),
                )
                logger.info("[Tool] Tushare fina_indicator 返回: ROE=%s 负债率=%s 毛利率=%s 净利同比=%s",
                           item["fin_roe"], item["fin_debt_ratio"], item["fin_grossprofit_margin"], item["fin_net_profit_yoy"])
        except Exception:
            logger.warning("[Tool] Tushare fina_indicator 失败 %s", ts_code)

        time.sleep(0.3)

        try:
            logger.info("[Tool] AKShare stock_news_em → %s", code)
            news_df = ak.stock_news_em(symbol=code)
            time.sleep(0.5)
            if news_df is not None and not news_df.empty:
                item["news_titles"] = news_df["新闻标题"].head(5).tolist()
                logger.info("[Tool] AKShare news 返回 %d 条新闻: %s",
                           len(item["news_titles"]), item["news_titles"][0] if item["news_titles"] else "")
        except Exception:
            logger.warning("[Tool] AKShare news 失败 %s", code)

        raw.append(item)
    return {"raw_data": raw}


def _llm_assess_company(state: CompanyState) -> dict:
    """LLM 逐股分析基本面 + 新闻，输出结构化 JSON。"""
    assessments = []
    for item in state.get("raw_data", []):
        fin = (
            f"ROE={item['fin_roe']}%, 负债率={item['fin_debt_ratio']}%, "
            f"毛利率={item['fin_grossprofit_margin']}%, 净利同比={item['fin_net_profit_yoy']}%"
        ) if item.get("fin_roe") is not None else "无财务数据"
        news = "; ".join(item.get("news_titles", [])) or "无"
        prompt = (
            f"股票 {item['code']} {item['name']}\n"
            f"财务指标：{fin}\n"
            f"近期新闻标题：{news}\n\n"
            f"分析这只股票的基本面风险，输出 JSON：\n"
            f'{{"risk_level": "aaa/aa/a/bb/cc", "risk_signals": [...], '
            f'"positives": [...], "summary": "..."}}\n\n'
            f"评级标准：\n"
            f"- aaa：ROE>15%、负债率<50%、毛利率>30%、利润正增长、无负面新闻\n"
            f"- aa：基本面健康，个别指标轻微偏弱\n"
            f"- a：存在1个明显风险点（如负债率偏高、利润下滑等）\n"
            f"- bb：多个风险点，或财务明显恶化\n"
            f"- cc：严重风险（ST、立案调查、ROE为负、重大亏损等）\n"
            f"注意：银行/金融行业负债率高是正常现象，不视为风险。"
        )
        try:
            from src.llm import chat_json
            logger.info("[LLM] Company Agent 分析 %s %s", item["code"], item["name"])
            logger.info("[LLM] Prompt:\n%s", prompt[:300])
            parsed = chat_json(
                "你是一个A股投研分析师。输出 JSON，不要 markdown 包裹。",
                prompt,
            )
            logger.info("[LLM] Company Agent 返回: risk_level=%s, signals=%s, summary=%s",
                       parsed.get("risk_level"), parsed.get("risk_signals"), parsed.get("summary", "")[:80])
        except Exception:
            parsed = {}
            logger.warning("[LLM] Company Agent 失败: %s", item["code"])

        assessments.append({
            **item,
            "risk_level": parsed.get("risk_level", "a"),
            "llm_risk_signals": parsed.get("risk_signals", []),
            "llm_positives": parsed.get("positives", []),
            "llm_summary": parsed.get("summary", "LLM 分析暂不可用"),
        })
    return {"company_assessments": assessments}


company_wf = StateGraph(CompanyState)
company_wf.add_node("collect_data", _collect_company_data)
company_wf.add_node("llm_assess", _llm_assess_company)
company_wf.set_entry_point("collect_data")
company_wf.add_edge("collect_data", "llm_assess")
company_wf.add_edge("llm_assess", END)
company_agent = company_wf.compile()


# ════════════════════════════════════════════════════════════════════
# Risk Agent — ST / 停牌 / 财务 / 新闻 综合风险研判
# ════════════════════════════════════════════════════════════════════

class RiskState(TypedDict):
    holdings: list[dict[str, Any]]
    st_codes: list[str]
    suspended_codes: list[str]
    risk_assessments: list[dict[str, Any]]
    error: str | None


# ════════════════════════════════════════════════════════════════════
# Strategy Agent — 综合策略建议
# ════════════════════════════════════════════════════════════════════


class StrategyState(TypedDict):
    market_sentiment: str
    holdings: list[dict[str, Any]]
    analysis: list[dict[str, Any]]
    strategy_result: dict[str, Any]
    error: str | None


def _generate_strategy(state: StrategyState) -> dict:
    """LLM 综合所有 Agent 输出，生成策略建议。"""
    sentiment = state.get("market_sentiment", "")[:200]
    analysis = state.get("analysis", [])
    holdings = state.get("holdings", [])

    # 构建持仓摘要
    lines = [f"市场情绪：{sentiment}"]
    lines.append(f"持仓 {len(holdings)} 只：")
    for a in analysis:
        sigs = "、".join(a.get("risk_signals", [])[:3]) or "无"
        pos = "、".join(a.get("positives", [])[:2]) or ""
        lines.append(
            f"- {a.get('code','')} {a.get('name','')} "
            f"评级={a.get('risk_level','')}, "
            f"ROE={a.get('fin_roe','?')}%, 负债率={a.get('fin_debt_ratio','?')}%"
            f"{', 风险:'+sigs if sigs != '无' else ''}"
            f"{', 亮点:'+pos if pos else ''}"
        )

    prompt = "\n".join(lines)
    prompt += (
        "\n\n请根据以上信息给出投资策略建议，包含：\n"
        "1. 市场判断（短期走势）\n"
        "2. 持仓建议（调仓方向）\n"
        "3. 风险提示\n"
        "4. 明日关注方向\n"
        "输出 JSON：\n"
        '{"market_judgment":"...","position_suggestions":"...",'
        '"risk_reminders":"...","tomorrow_focus":"...","overall_strategy":"..."}'
    )

    try:
        from src.llm import chat_json
        logger.info("[LLM] Strategy Agent 生成策略建议")
        result = chat_json(
            "你是一个A股策略分析师，综合市场、公司、风险数据给出建议。",
            prompt,
        )
        logger.info("[LLM] Strategy Agent: %s", result.get("overall_strategy", "")[:80])
    except Exception:
        logger.warning("[LLM] Strategy Agent 调用失败")
        result = {
            "market_judgment": "LLM 暂不可用",
            "position_suggestions": "维持现有持仓",
            "risk_reminders": "注意控制仓位",
            "tomorrow_focus": "关注市场变化",
            "overall_strategy": "观望",
        }

    return {"strategy_result": result}


strategy_wf = StateGraph(StrategyState)
strategy_wf.add_node("generate_strategy", _generate_strategy)
strategy_wf.set_entry_point("generate_strategy")
strategy_wf.add_edge("generate_strategy", END)
strategy_agent = strategy_wf.compile()


def _check_st_status(state: RiskState) -> dict:
    """检索全量股票名称，标记 ST / *ST / 退市。"""
    st_list: list[str] = []
    codes = [h["code"] for h in state.get("holdings", []) if h.get("code")]
    try:
        df = ak.stock_info_a_code_name()
        for _, row in df.iterrows():
            name = str(row.get("name", ""))
            code = str(row.get("code", ""))
            if code in codes and ("ST" in name or "退" in name):
                st_list.append(code)
                logger.info("[Risk] ST 检测 %s %s", code, name)
    except Exception:
        logger.warning("[Risk] ST 检测失败")
    return {"st_codes": st_list}


def _check_suspension(state: RiskState) -> dict:
    """查询当前是否停牌。

    过滤掉盘中短暂停牌（suspend_timing 有值），
    只检查全天停牌（suspend_timing 为 None）的最新记录是否为 S。
    """
    suspended: list[str] = []
    codes = [h["code"] for h in state.get("holdings", []) if h.get("code")]
    for code in codes:
        ts_code = _code_to_tscode(code)
        try:
            df = _ts_pro.query("suspend_d", ts_code=ts_code)
            if df is not None and not df.empty:
                # 过滤掉盘中短暂停牌（suspend_timing 非空，如 13:00-15:00）
                full = df[df["suspend_timing"].isna() | (df["suspend_timing"] == "")]
                if full.empty:
                    continue
                full = full.sort_values("trade_date", ascending=False)
                latest = full.iloc[0]
                if latest.get("suspend_type") == "S":
                    suspended.append(code)
                    logger.info("[Risk] 停牌检测 %s 当前停牌 (%s)", code, latest["trade_date"])
        except Exception:
            pass
        time.sleep(0.2)
    return {"suspended_codes": suspended}


def _assess_risks(state: RiskState) -> dict:
    """整合 ST / 停牌 标记到 risk_assessments。"""
    st_set = set(state.get("st_codes", []))
    susp_set = set(state.get("suspended_codes", []))
    assessments = []
    for h in state.get("holdings", []):
        code = h["code"]
        assessments.append({
            "code": code,
            "name": h.get("name", ""),
            "is_st": code in st_set,
            "is_suspended": code in susp_set,
        })
    return {"risk_assessments": assessments}


risk_wf = StateGraph(RiskState)
risk_wf.add_node("check_st", _check_st_status)
risk_wf.add_node("check_suspension", _check_suspension)
risk_wf.add_node("assess", _assess_risks)
risk_wf.set_entry_point("check_st")
risk_wf.add_edge("check_st", "check_suspension")
risk_wf.add_edge("check_suspension", "assess")
risk_wf.add_edge("assess", END)
risk_agent = risk_wf.compile()


# ════════════════════════════════════════════════════════════════════
# Report Agent — 生成投资日报
# ════════════════════════════════════════════════════════════════════


class ReportState(TypedDict):
    summary: dict[str, Any]
    analysis: list[dict[str, Any]]
    risk_signals: list[str]
    market_sentiment: str
    strategy_result: dict[str, Any]
    report_text: str
    error: str | None


def _generate_report(state: ReportState) -> dict:
    """LLM 综合所有数据生成日报。"""
    summary = state.get("summary", {})
    analysis = state.get("analysis", [])
    signals = state.get("risk_signals", [])
    sentiment = state.get("market_sentiment", "")[:200]
    strategy = state.get("strategy_result", {})

    total = summary.get("total_stocks", 0)
    lines = [f"市场情绪：{sentiment}"]
    lines.append(f"持仓 {total} 只：")
    for a in analysis:
        sigs = "、".join(a.get("risk_signals", [])[:2]) or "无"
        lines.append(
            f"- {a.get('code','')} {a.get('name','')} "
            f"[{a.get('risk_level','')}] "
            f"ROE={a.get('fin_roe','?')}% "
            f"{'风险:'+sigs if sigs!='无' else ''}"
        )
    if signals:
        lines.append(f"全局风险：{'；'.join(signals[:5])}")
    for k, v in strategy.items():
        if v:
            lines.append(f"{k}：{v}")

    prompt = "\n".join(lines) + (
        "\n\n根据以上数据生成一份今日投资日报，包含以下章节：\n"
        "## 今日市场\n## 持仓点评\n## 风险预警\n## 明日关注\n\n"
        "输出 JSON：\n"
        '{"report_title":"...","market_review":"...","portfolio_review":"...",'
        '"risk_warnings":"...","tomorrow_focus":"...","full_report":"..."}'
    )
    try:
        from src.llm import chat_json
        logger.info("[LLM] Report Agent 生成日报")
        result = chat_json(
            "你是一个A股投资日报编辑，生成专业简洁的日报。",
            prompt,
        )
        logger.info("[LLM] Report Agent 完成")
    except Exception:
        logger.warning("[LLM] Report Agent 失败")
        result = {"full_report": "日报生成暂不可用", "report_title": ""}

    return {"report_text": result.get("full_report", ""), "report_data": result}


report_wf = StateGraph(ReportState)
report_wf.add_node("generate_report", _generate_report)
report_wf.set_entry_point("generate_report")
report_wf.add_edge("generate_report", END)
report_agent = report_wf.compile()


# ════════════════════════════════════════════════════════════════════
# 主图 — 合并 Market + Company + Risk + Strategy，生成最终评级
# ════════════════════════════════════════════════════════════════════

def run_market_agent(state: PortfolioState) -> dict:
    """Node: 调用 Market Agent 子图。"""
    m_state: MarketState = {"holdings": state.get("holdings", [])}
    r = market_agent.invoke(m_state)
    return {
        "market_indices": r.get("market_indices", {}),
        "limit_up_codes": r.get("limit_up_codes", []),
        "market_assessments": r.get("market_assessments", []),
        "sentiment_analysis": r.get("sentiment_analysis", ""),
    }


def run_company_agent(state: PortfolioState) -> dict:
    """Node: 调用 Company Agent 子图。"""
    c_state: CompanyState = {"holdings": state.get("holdings", [])}
    r = company_agent.invoke(c_state)
    return {"company_assessments": r.get("company_assessments", [])}


def run_risk_agent(state: PortfolioState) -> dict:
    """Node: 调用 Risk Agent 子图。"""
    r_state: RiskState = {"holdings": state.get("holdings", [])}
    r = risk_agent.invoke(r_state)
    return {
        "st_codes": r.get("st_codes", []),
        "suspended_codes": r.get("suspended_codes", []),
        "risk_assessments": r.get("risk_assessments", []),
    }


def merge_analysis(state: PortfolioState) -> dict:
    """Node: 合并 Market + Company + Risk 结果（屏障节点，等待三路并行全部完成）。"""
    market = state.get("market_assessments")
    company = state.get("company_assessments")
    risk_list = state.get("risk_assessments")

    # 屏障：任一结果还未就绪则不执行
    if not market or not company or risk_list is None:
        return {}

    st_set = set(state.get("st_codes", []))
    susp_set = set(state.get("suspended_codes", []))

    analysis, signals = [], []
    for ma in market:
        code = ma["code"]
        ca = next((c for c in company if c["code"] == code), {})
        merged = {**ca, "is_limit_up": ma.get("is_limit_up", False)}

        all_sigs = list(ca.get("llm_risk_signals", []))
        if code in st_set:
            all_sigs.append("ST / *ST")
        if code in susp_set:
            all_sigs.append("停牌")
        if not all_sigs:
            if ca.get("fin_roe") is not None and ca["fin_roe"] < 0:
                all_sigs.append("ROE 为负")
            if ca.get("fin_debt_ratio") is not None and ca["fin_debt_ratio"] > 70:
                all_sigs.append(f"负债率 {ca['fin_debt_ratio']:.0f}%")
            if ca.get("fin_net_profit_yoy") is not None and ca["fin_net_profit_yoy"] < -50:
                all_sigs.append("净利大幅下滑")
        merged["risk_signals"] = all_sigs
        merged["positives"] = list(ca.get("llm_positives", []))
        if ma.get("is_limit_up"):
            merged["positives"].append("今日涨停")

        _valid = ("aaa", "aa", "a", "bb", "cc")
        if code in st_set:
            merged["risk_level"] = "cc"
        elif code in susp_set:
            merged["risk_level"] = "bb"
        else:
            llm_level = ca.get("risk_level", "").lower()
            if llm_level in _valid:
                merged["risk_level"] = llm_level
            else:
                n = len(all_sigs)
                merged["risk_level"] = "aa" if n < 1 else ("a" if n < 2 else ("bb" if n < 3 else "cc"))
        analysis.append(merged)
        signals.extend(all_sigs)

    return {
        "analysis": analysis,
        "risk_signals": list(dict.fromkeys(signals)),
        "market_sentiment": state.get("sentiment_analysis", ""),
    }


def run_strategy_agent(state: PortfolioState) -> dict:
    """Node: 调用 Strategy Agent 子图。"""
    s_state: StrategyState = {
        "market_sentiment": state.get("sentiment_analysis", ""),
        "holdings": state.get("holdings", []),
        "analysis": state.get("analysis", []),
    }
    r = strategy_agent.invoke(s_state)
    return {"strategy_result": r.get("strategy_result", {})}


def run_report_agent(state: PortfolioState) -> dict:
    """Node: 调用 Report Agent 生成日报。"""
    r_state: ReportState = {
        "summary": state.get("summary", {}),
        "analysis": state.get("analysis", []),
        "risk_signals": state.get("risk_signals", []),
        "market_sentiment": state.get("sentiment_analysis", ""),
        "strategy_result": state.get("strategy_result", {}),
    }
    result = report_agent.invoke(r_state)
    return {"report_text": result.get("report_text", "")}


def router(state: PortfolioState) -> str:
    return "error_end" if state.get("error") else "continue"


def merge_router(state: PortfolioState) -> str:
    """屏障路由：三路并行全部完成后才继续。"""
    if state.get("market_assessments") and state.get("company_assessments") and state.get("risk_assessments") is not None:
        return "ready"
    return "wait"


wg = StateGraph(PortfolioState)
for name in ("parse_csv", "enrich_holdings", "generate_summary",
             "run_market_agent", "run_company_agent",
             "run_risk_agent", "merge_analysis",
             "run_strategy_agent", "run_report_agent"):
    wg.add_node(name, locals()[name])
wg.set_entry_point("parse_csv")

# 串行：解析 → 补名称
wg.add_conditional_edges("parse_csv", router, {"continue": "enrich_holdings", "error_end": END})
wg.add_conditional_edges("enrich_holdings", router, {"continue": "run_market_agent", "error_end": END})

# 并行：market / company / risk 三条分支同时跑
wg.add_edge("run_market_agent", "merge_analysis")
wg.add_edge("run_company_agent", "merge_analysis")
wg.add_edge("run_risk_agent", "merge_analysis")

# 屏障：等待三路全部完成才放行
wg.add_conditional_edges("merge_analysis", merge_router,
                          {"ready": "run_strategy_agent", "wait": END})

# 串行：策略 → 日报 → 汇总
wg.add_conditional_edges("run_strategy_agent", router, {"continue": "run_report_agent", "error_end": END})
wg.add_conditional_edges("run_report_agent", router, {"continue": "generate_summary", "error_end": END})
wg.add_edge("generate_summary", END)

portfolio_agent = wg.compile()


# ════════════════════════════════════════════════════════════════════
# SSE 流式分析
# ════════════════════════════════════════════════════════════════════


def _sse(event: str, data: Any) -> str:
    """格式化为 SSE 事件字符串。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _run_analysis_stream(csv_text: str):
    """逐阶段执行分析，SSE 只推送进度事件，完成后发给前端 result_id。"""
    state: PortfolioState = {"csv_content": csv_text}

    # ── 解析 CSV ──────────────────────────────────────────────────
    r = parse_csv(state)
    if r.get("error"):
        yield _sse("error", {"message": r["error"]})
        return
    state.update(r)

    # ── 补充名称 ───────────────────────────────────────────────────
    r = enrich_holdings(state)
    state.update(r)
    holdings = state.get("holdings", [])
    total = len(holdings)

    yield _sse("parsed", {
        "total": total,
        "holdings": [{"code": h["code"], "name": h["name"]} for h in holdings],
    })

    # ── 三路并行：Market + Company + Risk ─────────────────────────
    yield _sse("progress", {"label": "📈 读取大盘指数 & 涨停数据...", "detail": ""})
    yield _sse("progress", {"label": "📊 分析个股财务数据 & 新闻...", "detail": ""})
    yield _sse("progress", {"label": "🔍 排查 ST / 停牌风险...", "detail": ""})

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_market = pool.submit(market_agent.invoke, {"holdings": holdings})
        fut_company = pool.submit(company_agent.invoke, {"holdings": holdings})
        fut_risk = pool.submit(risk_agent.invoke, {"holdings": holdings})

        fut_map = {fut_market: "market", fut_company: "company", fut_risk: "risk"}

        for fut in as_completed([fut_market, fut_company, fut_risk]):
            tp = fut_map[fut]
            if tp == "market":
                mr = fut.result()
                state["market_indices"] = mr.get("market_indices", {})
                state["limit_up_codes"] = mr.get("limit_up_codes", [])
                state["market_assessments"] = mr.get("market_assessments", [])
                idx = mr.get("market_indices", {})
                idx_str = "; ".join(f"{k} {v.get('pct_chg','?')}%" for k, v in idx.items())
                limit_n = len(mr.get("limit_up_codes", []))
                yield _sse("progress", {"label": f"📈 大盘: {idx_str} | 涨停 {limit_n} 只"})
            elif tp == "company":
                ca = fut.result()
                state["company_assessments"] = ca.get("company_assessments", [])
                for a in (ca.get("company_assessments") or []):
                    yield _sse("progress", {"label": f"📊 {a.get('code','')} {a.get('name','')}: ROE={a.get('fin_roe','?')}% 负债率={a.get('fin_debt_ratio','?')}% 评级={a.get('risk_level','?')}"})
            elif tp == "risk":
                rr = fut.result()
                state["st_codes"] = rr.get("st_codes", [])
                state["suspended_codes"] = rr.get("suspended_codes", [])
                state["risk_assessments"] = rr.get("risk_assessments", [])
                for h in holdings:
                    c = h["code"]
                    msgs = []
                    if c in rr.get("st_codes", []): msgs.append("⚠️ ST")
                    if c in rr.get("suspended_codes", []): msgs.append("⏸️ 停牌")
                    if msgs:
                        yield _sse("progress", {"label": f"🔍 {c} {h.get('name','')}: {' '.join(msgs)}"})
                yield _sse("progress", {"label": "🔍 风险排查完成"})

    # ── Merge：合并评级 ─────────────────────────────────────────────
    yield _sse("progress", {"label": "🔄 综合评级中..."})
    merged = merge_analysis(state)
    state.update(merged)
    yield _sse("progress", {"label": "✅ 评级合并完成", "current": 0, "total": 1})

    # ── Strategy Agent：策略建议 ──────────────────────────────────
    yield _sse("progress", {"label": "🧠 生成策略建议...", "current": 0, "total": 1})
    try:
        s_state: StrategyState = {
            "market_sentiment": state.get("sentiment_analysis", ""),
            "holdings": holdings,
            "analysis": state.get("analysis", []),
        }
        sr = strategy_agent.invoke(s_state)
        state["strategy_result"] = sr.get("strategy_result", {})
    except Exception:
        logger.warning("[Strategy] Agent 失败", exc_info=True)

    # ── Report Agent：生成日报 ─────────────────────────────────
    yield _sse("progress", {"phase": "report", "label": "📝 生成投资日报", "current": 0, "total": 1})
    try:
        rpt_s: ReportState = {
            "summary": state.get("summary", {}),
            "analysis": state.get("analysis", []),
            "risk_signals": state.get("risk_signals", []),
            "market_sentiment": state.get("sentiment_analysis", ""),
            "strategy_result": state.get("strategy_result", {}),
        }
        rpt = report_agent.invoke(rpt_s)
        state["report_text"] = rpt.get("report_text", "")
    except Exception:
        logger.warning("[Report] Agent 失败", exc_info=True)

    # ── 写入 Memory + RAG ────────────────────────────────────────
    try:
        memory.save_portfolio(holdings)
        for a in state.get("analysis", []):
            memory.save_stock_note(
                code=a.get("code", ""),
                name=a.get("name", ""),
                risk_level=a.get("risk_level", "a"),
                summary=a.get("llm_summary", ""),
                fin_roe=a.get("fin_roe"),
                fin_debt_ratio=a.get("fin_debt_ratio"),
                fin_grossprofit_margin=a.get("fin_grossprofit_margin"),
                fin_net_profit_yoy=a.get("fin_net_profit_yoy"),
                market_sentiment=state.get("sentiment_analysis", ""),
            )
            # 同时写入 RAG 向量库
            try:
                rag.add_analysis_result(a)
            except Exception:
                logger.warning("[RAG] 写入失败", exc_info=True)
        rag.save()
    except Exception:
        logger.warning("[Memory/RAG] 写入失败", exc_info=True)

    # ── 汇总 ───────────────────────────────────────────────────────
    r = generate_summary(state)
    state.update(r)

    # ── 缓存结果，发给前端 result_id ──────────────────────────────
    result_id = uuid.uuid4().hex[:12]
    _result_cache[result_id] = {
        "summary": state.get("summary", {}),
        "holdings": holdings,
        "analysis": state.get("analysis", []),
        "risk_signals": state.get("risk_signals", []),
        "market_sentiment": state.get("market_sentiment", "neutral"),
        "market_indices": state.get("market_indices", {}),
        "strategy_result": state.get("strategy_result", {}),
        "report_text": state.get("report_text", ""),
    }

    analysis_list = state.get("analysis", [])
    yield _sse("complete", {
        "result_id": result_id,
        "summary": {
            "total": state.get("summary", {}).get("total_stocks", 0),
            "aaa": sum(1 for a in analysis_list if a.get("risk_level") == "aaa"),
            "aa": sum(1 for a in analysis_list if a.get("risk_level") == "aa"),
            "a": sum(1 for a in analysis_list if a.get("risk_level") == "a"),
            "bb": sum(1 for a in analysis_list if a.get("risk_level") == "bb"),
            "cc": sum(1 for a in analysis_list if a.get("risk_level") == "cc"),
            "stocks": [{"code": h["code"], "name": h["name"]} for h in holdings],
        },
    })


# ════════════════════════════════════════════════════════════════════
# Flask Routes
# ════════════════════════════════════════════════════════════════════


# ── 股票名称 → 代码 缓存 ─────────────────────────────────────────────
_stock_name_cache: dict[str, str] | None = None


def _build_name_code_map() -> dict[str, str]:
    global _stock_name_cache
    if _stock_name_cache is not None:
        return _stock_name_cache
    try:
        df = ak.stock_info_a_code_name()
        _stock_name_cache = dict(zip(df["name"], df["code"]))
        return _stock_name_cache
    except Exception:
        return {}


def _search_stock_code(query: str) -> str | None:
    """根据用户输入模糊匹配股票代码。"""
    mapping = _build_name_code_map()
    # 精确匹配
    if query in mapping:
        return mapping[query]
    # 模糊匹配
    for name, code in mapping.items():
        if query in name or name in query:
            return code
    return None


# ════════════════════════════════════════════════════════════════════
# 聊天 SSE 端点
# ════════════════════════════════════════════════════════════════════


@app.route("/chat", methods=["POST"])
def chat_api():
    """聊天接口：接收用户消息，流式返回 LLM 回复。"""
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return {"error": "消息不能为空"}, 400

    def generate():
        # 1. 用 LLM 判断意图 & 提取股票名
        intent_prompt = (
            f"用户说：{user_msg}\n\n"
            f"判断意图，仅输出 JSON：\n"
            f'{{"type": "stock_query"/"strategy"/"chat", '
            f'"stock_name": "股票名称或null"}}'
        )
        try:
            from src.llm import chat_json
            intent = chat_json("你是一个股票查询意图识别助手。", intent_prompt)
        except Exception:
            intent = {"type": "chat", "stock_name": None}

        intent_type = intent.get("type", "chat")
        stock_code = None
        if intent_type == "stock_query":
            stock_name = intent.get("stock_name") or user_msg
            stock_code = _search_stock_code(stock_name)

        NL = "\n"
        context_parts = []

        yield f"data: {json.dumps({'token': '🔄 分析中'})}\n\n"

        if stock_code:
            _qname = intent.get("stock_name", stock_code) or stock_code
            yield f"data: {json.dumps({'token': f' 🔍 {_qname}'})}\n\n"

            # 获取股票名称
            name_map = _build_stock_name_map()
            stock_name = name_map.get(stock_code, stock_code)

            # 查财务 + 新闻 (Company Agent)
            yield f"data: {json.dumps({'token': ' 📊'})}\n\n"
            try:
                ca = company_agent.invoke({"holdings": [{"code": stock_code, "name": stock_name}]})
                comp = ca.get("company_assessments", [{}])[0]
                if comp.get("fin_roe") is not None:
                    context_parts.append(
                        f"财务数据：ROE={comp['fin_roe']}%, "
                        f"负债率={comp['fin_debt_ratio']}%, "
                        f"毛利率={comp['fin_grossprofit_margin']}%, "
                        f"净利同比={comp['fin_net_profit_yoy']}%"
                    )
                if comp.get("news_titles"):
                    context_parts.append(f"近期新闻：{'；'.join(comp['news_titles'][:3])}")
            except Exception as e:
                context_parts.append(f"财务数据获取异常：{e}")

            time.sleep(0.3)

            # 查市场 (Market Agent)
            yield f"data: {json.dumps({'token': ' 📈'})}\n\n"
            try:
                mr = market_agent.invoke({"holdings": [{"code": stock_code, "name": stock_name}]})
                idx = mr.get("market_indices", {})
                if idx:
                    parts = [f"{k} {v.get('pct_chg', '?')}%" for k, v in idx.items()]
                    context_parts.append(f"大盘表现：{' | '.join(parts)}")
                limit_codes = mr.get("limit_up_codes", [])
                if stock_code in limit_codes:
                    context_parts.append("📈 该股今日涨停")
            except Exception:
                pass

            # 从 Memory 读取历史分析记录
            try:
                notes = memory.get_stock_notes(stock_code)
                if notes:
                    last = notes[0]
                    context_parts.append(
                        f"前次分析 ({last.get('created_at','?')})："
                        f"评级={last['risk_level']}, {last.get('analysis_summary','')}"
                    )
            except Exception:
                pass

        # RAG 检索相关历史分析
        if stock_code:
            try:
                rag_results = rag.search(f"{stock_code} {stock_name}", top_k=3)
                for r in rag_results:
                    if r.get("score", 0) > 0.3:
                        context_parts.append(
                            f"[历史] {r.get('company','')}: {r.get('content','')[:120]}"
                        )
            except Exception:
                pass

        # 从 Memory 读取用户关注偏好
        # ── Strategy 意图：从 memory 加载持仓，跑全部分析 ────────────
        if intent_type == "strategy" and not stock_code:
            try:
                portfolios = memory.get_portfolio_history(limit=30)
                codes = list(dict.fromkeys(p["stock_code"] for p in portfolios if p.get("stock_code")))
                if codes:
                    yield f"data: {json.dumps({'token': f'📊 从历史记录加载 {len(codes)} 只持仓，开始分析...{NL}{NL}'})}\n\n"
                    name_map = _build_stock_name_map()
                    holdings = [{"code": c, "name": name_map.get(c, c)} for c in codes]
                    yield _sse("msg", {"token": "📈 拉取大盘数据 & 涨停池..."})
                    mr = market_agent.invoke({"holdings": holdings})
                    idx = mr.get("market_indices", {})
                    idx_str = "; ".join(f"{k} {v.get('pct_chg','?')}%" for k, v in idx.items())
                    limit_n = len(mr.get("limit_up_codes") or [])
                    yield _sse("msg", {"token": f"✅ 大盘: {idx_str} | 涨停 {limit_n} 只"})

                    yield _sse("msg", {"token": "📊 分析个股财务数据 & 新闻..."})
                    ca = company_agent.invoke({"holdings": holdings})
                    for a in (ca.get("company_assessments") or []):
                        c = a.get("code",""); n = a.get("name","")
                        r = a.get("fin_roe","?"); d = a.get("fin_debt_ratio","?")
                        lv = a.get("risk_level","?")
                        yield _sse("msg", {"token": f"📊 {c} {n}: ROE={r}% 负债率={d}% 评级={lv}"})

                    yield _sse("msg", {"token": "🔍 排查 ST / 停牌风险..."})
                    rr = risk_agent.invoke({"holdings": holdings})
                    for h in holdings:
                        c = h["code"]; nm = h.get("name","")
                        risk_msgs = []
                        if c in (rr.get("st_codes") or []): risk_msgs.append("ST")
                        if c in (rr.get("suspended_codes") or []): risk_msgs.append("停牌")
                        if risk_msgs:
                            yield _sse("msg", {"token": f"🔍 {c} {nm}: {' '.join(risk_msgs)}"})
                    yield f"data: {json.dumps({'token': f'✅ 风险排查完成{NL}'})}\n\n"

                    # 汇编 analysis
                    analysis = []
                    for h in holdings:
                        c = h["code"]
                        comp = next((x for x in (ca.get("company_assessments") or []) if x["code"] == c), {})
                        risk = next((x for x in (rr.get("risk_assessments") or []) if x["code"] == c), {})
                        st_set = set(rr.get("st_codes", []))
                        sigs = list(comp.get("llm_risk_signals", []))
                        if c in st_set:
                            sigs.append("ST/*ST")
                        analysis.append({
                            "code": c, "name": h["name"],
                            "risk_level": comp.get("risk_level", "a"),
                            "risk_signals": sigs,
                            "fin_roe": comp.get("fin_roe"),
                            "fin_debt_ratio": comp.get("fin_debt_ratio"),
                            "fin_grossprofit_margin": comp.get("fin_grossprofit_margin"),
                            "fin_net_profit_yoy": comp.get("fin_net_profit_yoy"),
                        })
                    yield f"data: {json.dumps({'token': ' 🧠'})}\n\n"
                    # 跑 Strategy Agent
                    sr = strategy_agent.invoke({
                        "market_sentiment": mr.get("sentiment_analysis", ""),
                        "holdings": holdings,
                        "analysis": analysis,
                    })
                    strat = sr.get("strategy_result", {})
                    for k, v in strat.items():
                        if v:
                            context_parts.append(f"{k}: {v}")
            except Exception:
                logger.warning("[Chat] Strategy 分析失败", exc_info=True)

        try:
            interests = memory.get_interests(limit=5)
            if interests:
                ctx = extractor.format_interests_for_prompt(interests)
                if ctx:
                    context_parts.append(ctx)
        except Exception:
            pass

        yield f"data: {json.dumps({'token': f' ✨{NL}{NL}'})}\n\n"

        # 3. 构建 LLM 系统提示 + 用户消息
        system = "你是一个A股投研分析师。回答简洁专业，用中文。"
        if (stock_code or intent_type == "strategy") and context_parts:
            _name = stock_name if stock_code else "持仓组合"
            _code = stock_code or ""
            user_prompt = (
                f"用户问：{user_msg}\n\n"
                f"股票 {_name} ({_code}) 的相关数据：\n" +
                "\n".join(context_parts) +
                "\n\n请根据以上数据回答用户的问题。"
            )
        else:
            user_prompt = user_msg

        # 4. 流式调用千问
        try:
            logger.info("[LLM] Chat API 用户问题: %s", user_msg[:200])
            if stock_code:
                logger.info("[LLM] Chat API 匹配股票: %s %s, 数据:\n%s", stock_code, stock_name, "\n".join(context_parts))
            from openai import OpenAI
            from src.config import llm_config
            _lc = llm_config()
            client = OpenAI(
                api_key=_lc["api_key"],
                base_url=_lc["base_url"],
            )
            stream = client.chat.completions.create(
                model=_lc["model"],
                temperature=0.1,
                max_tokens=2048,
                stream=True,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
            )
            full_reply: list[str] = []
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_reply.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # 写入 Memory — 提取用户关注偏好
            try:
                interests = extractor.extract_interests_from_conversation(
                    user_msg, "".join(full_reply),
                )
                for item in interests:
                    memory.save_interest(
                        category=item["category"],
                        content=item["content"],
                        confidence=item.get("confidence", 0.8),
                        source="chat",
                    )
                if interests:
                    logger.info("[Memory] 提取 %d 条用户关注偏好", len(interests))
            except Exception:
                pass
        except Exception as e:
            yield f"data: {json.dumps({'token': f'❌ LLM 调用失败: {e}'})}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Stream SSE events: parsed → progress×N → complete(含渲染好的 HTML)。"""
    if "file" not in request.files:
        return render_template("index.html", error="请选择文件")
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="文件名为空")
    if not file.filename.endswith(".csv"):
        return render_template("index.html", error="仅支持 CSV 文件")

    csv_text = file.read().decode("utf-8-sig")

    return Response(
        _run_analysis_stream(csv_text),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/result/<result_id>")
def show_result(result_id: str):
    """分析完成后，前端跳转到这里展示结果页。"""
    data = _result_cache.pop(result_id, None)
    if not data:
        return render_template("index.html", error="结果已过期，请重新上传")

    return render_template(
        "result.html",
        summary=data["summary"],
        holdings=data["holdings"],
        analysis=data["analysis"],
        risk_signals=data["risk_signals"],
        market_sentiment=data.get("market_sentiment", "neutral"),
        market_indices=data.get("market_indices", {}),
        strategy_result=data.get("strategy_result", {}),
        report_text=data.get("report_text", ""),
    )


# ════════════════════════════════════════════════════════════════════
# Memory API
# ════════════════════════════════════════════════════════════════════


@app.route("/api/memory/interests")
def api_memory_interests():
    """返回用户关注偏好。"""
    try:
        items = memory.get_interests(limit=30)
        return {"items": items, "total": len(items)}
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/api/memory/history")
def api_memory_history():
    """返回持仓历史和最近分析记录。"""
    try:
        portfolios = memory.get_portfolio_history(limit=30)
        notes_list = []
        seen = set()
        for h in portfolios[:10]:
            code = h.get("stock_code")
            if code and code not in seen:
                seen.add(code)
                notes = memory.get_stock_notes(code, limit=1)
                if notes:
                    notes_list.append(notes[0])
        return {"portfolios": portfolios, "recent_notes": notes_list}
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/api/memory/clear", methods=["POST"])
def api_memory_clear():
    """清空数据库。"""
    try:
        db_path = os.path.join(os.path.dirname(__file__), "data", "memory.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            from memory.schema import init_memory_database
            init_memory_database(db_path)
            logger.info("[Memory] 数据库已清空")
            return {"ok": True}
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}, 500


# ════════════════════════════════════════════════════════════════════
# Helper
# ════════════════════════════════════════════════════════════════════


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_pct(v: Any, max_val: float = 100) -> float | None:
    """安全的百分比转换，超合理范围则返回 None。"""
    val = _safe_float(v)
    if val is None:
        return None
    # Tushare 偶有返回原始金额而非比率的脏数据，过滤掉
    
    if abs(val) > max_val * 10:   # 远超合理范围 → 脏数据
        return None
    return val


if __name__ == "__main__":
    setup_logging()
    app.run(debug=True, port=5001, threaded=True)
