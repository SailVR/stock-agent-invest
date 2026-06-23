"""SSE workflow for portfolio CSV analysis."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from memory.long_term_memory import LongTermMemory
from src.agents import (
    PortfolioState,
    ReportState,
    StrategyState,
    company_agent,
    enrich_holdings,
    generate_summary,
    market_agent,
    merge_analysis,
    parse_csv,
    report_agent,
    risk_agent,
    strategy_agent,
)
from src.logs import get_logger

logger = get_logger(__name__)


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def run_analysis_stream(
    csv_text: str,
    memory: LongTermMemory,
    get_rag: Callable[[], Any],
    result_cache: dict[str, dict],
):
    """Run portfolio analysis and stream progress as Server-Sent Events."""
    state: PortfolioState = {"csv_content": csv_text}

    result = parse_csv(state)
    if result.get("error"):
        yield sse("error", {"message": result["error"]})
        return
    state.update(result)

    result = enrich_holdings(state)
    state.update(result)
    holdings = state.get("holdings", [])

    yield sse("parsed", {
        "total": len(holdings),
        "holdings": [{"code": h["code"], "name": h["name"]} for h in holdings],
    })

    yield sse("progress", {"label": "📈 读取大盘指数 & 涨停数据...", "detail": ""})
    yield sse("progress", {"label": "📊 分析个股财务数据 & 新闻...", "detail": ""})
    yield sse("progress", {"label": "🔍 排查 ST / 停牌风险...", "detail": ""})

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(market_agent.invoke, {"holdings": holdings}): "market",
            pool.submit(company_agent.invoke, {"holdings": holdings}): "company",
            pool.submit(risk_agent.invoke, {"holdings": holdings}): "risk",
        }

        for fut in as_completed(futures):
            tp = futures[fut]
            if tp == "market":
                mr = fut.result()
                state["market_indices"] = mr.get("market_indices", {})
                state["limit_up_codes"] = mr.get("limit_up_codes", [])
                state["market_assessments"] = mr.get("market_assessments", [])
                state["sentiment_analysis"] = mr.get("sentiment_analysis", "")
                idx = mr.get("market_indices", {})
                idx_str = "; ".join(f"{k} {v.get('pct_chg','?')}%" for k, v in idx.items())
                limit_n = len(mr.get("limit_up_codes", []))
                yield sse("progress", {"label": f"📈 大盘: {idx_str} | 涨停 {limit_n} 只"})
            elif tp == "company":
                ca = fut.result()
                state["company_assessments"] = ca.get("company_assessments", [])
                for item in (ca.get("company_assessments") or []):
                    yield sse("progress", {
                        "label": (
                            f"📊 {item.get('code','')} {item.get('name','')}: "
                            f"ROE={item.get('fin_roe','?')}% "
                            f"负债率={item.get('fin_debt_ratio','?')}% "
                            f"评级={item.get('risk_level','?')}"
                        )
                    })
            elif tp == "risk":
                rr = fut.result()
                state["st_codes"] = rr.get("st_codes", [])
                state["suspended_codes"] = rr.get("suspended_codes", [])
                state["risk_assessments"] = rr.get("risk_assessments", [])
                for h in holdings:
                    code = h["code"]
                    msgs = []
                    if code in rr.get("st_codes", []):
                        msgs.append("⚠️ ST")
                    if code in rr.get("suspended_codes", []):
                        msgs.append("⏸️ 停牌")
                    if msgs:
                        yield sse("progress", {"label": f"🔍 {code} {h.get('name','')}: {' '.join(msgs)}"})
                yield sse("progress", {"label": "🔍 风险排查完成"})

    yield sse("progress", {"label": "🔄 综合评级中..."})
    state.update(merge_analysis(state))
    yield sse("progress", {"label": "✅ 评级合并完成", "current": 0, "total": 1})

    yield sse("progress", {"label": "🧠 生成策略建议...", "current": 0, "total": 1})
    try:
        strategy_state: StrategyState = {
            "market_sentiment": state.get("sentiment_analysis", ""),
            "holdings": holdings,
            "analysis": state.get("analysis", []),
        }
        sr = strategy_agent.invoke(strategy_state)
        state["strategy_result"] = sr.get("strategy_result", {})
    except Exception:
        logger.warning("[Strategy] Agent 失败", exc_info=True)

    yield sse("progress", {"phase": "report", "label": "📝 生成投资日报", "current": 0, "total": 1})
    try:
        report_state: ReportState = {
            "summary": state.get("summary", {}),
            "analysis": state.get("analysis", []),
            "risk_signals": state.get("risk_signals", []),
            "market_sentiment": state.get("sentiment_analysis", ""),
            "strategy_result": state.get("strategy_result", {}),
        }
        rpt = report_agent.invoke(report_state)
        state["report_text"] = rpt.get("report_text", "")
    except Exception:
        logger.warning("[Report] Agent 失败", exc_info=True)

    try:
        memory.save_portfolio(holdings)
        analysis_items = state.get("analysis", [])
        rag = get_rag() if analysis_items else None
        for item in analysis_items:
            memory.save_stock_note(
                code=item.get("code", ""),
                name=item.get("name", ""),
                risk_level=item.get("risk_level", "a"),
                summary=item.get("llm_summary", ""),
                fin_roe=item.get("fin_roe"),
                fin_debt_ratio=item.get("fin_debt_ratio"),
                fin_grossprofit_margin=item.get("fin_grossprofit_margin"),
                fin_net_profit_yoy=item.get("fin_net_profit_yoy"),
                market_sentiment=state.get("sentiment_analysis", ""),
            )
            try:
                rag.add_analysis_result(item)
            except Exception:
                logger.warning("[RAG] 写入失败", exc_info=True)
        if rag is not None:
            rag.save()
    except Exception:
        logger.warning("[Memory/RAG] 写入失败", exc_info=True)

    state.update(generate_summary(state))

    result_id = uuid.uuid4().hex[:12]
    result_cache[result_id] = {
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
    yield sse("complete", {
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
