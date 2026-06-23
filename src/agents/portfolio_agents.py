"""LangGraph agent definitions for portfolio analysis."""

from __future__ import annotations

import csv
import io
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.llm import chat
from src.logs import get_logger
from src.tools.company_tools import fetch_stock_news, get_financial_indicators
from src.tools.market_tools import get_limit_up_stocks, get_market_index
from src.tools.stock_tools import (
    code_to_tscode,
    get_st_codes,
    get_stock_name_map,
    get_suspended_codes,
)

logger = get_logger(__name__)


class PortfolioState(TypedDict):
    csv_content: str
    holdings: list[dict[str, Any]]
    summary: dict[str, Any]
    analysis: list[dict[str, Any]]
    risk_signals: list[str]
    strategy_result: dict[str, Any]
    report_text: str
    error: str | None


def build_stock_name_map() -> dict[str, str]:
    return get_stock_name_map()


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
    name_map = build_stock_name_map()
    enriched = [{**h, "name": name_map.get(h["code"], "未知")} for h in holdings]
    logger.info("Enriched %d holdings with names.", len(enriched))
    return {"holdings": enriched}


def generate_summary(state: PortfolioState) -> dict:
    holdings = state["holdings"]
    prey_vals = [h["prey"] for h in holdings if h.get("prey") is not None]
    summary: dict[str, Any] = {"total_stocks": len(holdings)}
    if prey_vals:
        summary.update(
            prey_mean=round(sum(prey_vals) / len(prey_vals), 4),
            prey_max=max(prey_vals),
            prey_min=min(prey_vals),
        )
    return {"summary": summary}


class MarketState(TypedDict):
    holdings: list[dict[str, Any]]
    market_indices: dict[str, Any]
    limit_up_codes: list[str]
    market_assessments: list[dict[str, Any]]
    sentiment_analysis: str
    error: str | None


def _fetch_market_data(state: MarketState) -> dict:
    symbols = {
        "上证指数": "000001.SH",
        "深证成指": "399001.SZ",
        "创业板指": "399006.SZ",
    }
    indices = get_market_index(symbols=symbols)
    limit_codes = [
        str(item.get("代码", "")).zfill(6)
        for item in get_limit_up_stocks()
        if item.get("代码")
    ]
    return {"market_indices": indices, "limit_up_codes": limit_codes}


def _llm_assess_market(state: MarketState) -> dict:
    indices = state.get("market_indices", {})
    summary = "; ".join(f"{k} 涨跌幅 {v.get('pct_chg', '?')}%" for k, v in indices.items())
    limit_cnt = len(state.get("limit_up_codes", []))
    prompt = (
        f"今日市场数据：{summary}；涨停 {limit_cnt} 只。\n"
        f"请分析当前市场情绪，给出判断（strong / neutral / weak）和简短理由。"
    )
    try:
        logger.info("[LLM] Market Agent 分析市场情绪")
        analysis = chat("你是一个A股市场分析专家，请根据指数和涨停数据判断市场情绪。", prompt)
        logger.info("[LLM] Market Agent 返回: %s", analysis[:200])
    except Exception as e:
        logger.warning("Market Agent LLM 调用失败: %s", e)
        analysis = f"市场数据：{summary}，涨停 {limit_cnt} 只"
    return {"sentiment_analysis": analysis}


def _market_assess_limit_up(state: MarketState) -> dict:
    return {
        "market_assessments": [
            {"code": h["code"], "name": h["name"], "is_limit_up": h["code"] in state.get("limit_up_codes", [])}
            for h in state.get("holdings", [])
        ]
    }


market_wf = StateGraph(MarketState)
market_wf.add_node("fetch_market_data", _fetch_market_data)
market_wf.add_node("assess_limit_up", _market_assess_limit_up)
market_wf.add_node("llm_assess", _llm_assess_market)
market_wf.set_entry_point("fetch_market_data")
market_wf.add_edge("fetch_market_data", "assess_limit_up")
market_wf.add_edge("assess_limit_up", "llm_assess")
market_wf.add_edge("llm_assess", END)
market_agent = market_wf.compile()


class CompanyState(TypedDict):
    holdings: list[dict[str, Any]]
    raw_data: list[dict[str, Any]]
    company_assessments: list[dict[str, Any]]
    error: str | None


def _collect_company_data(state: CompanyState) -> dict:
    raw = []
    for h in state.get("holdings", []):
        code, name = h.get("code", ""), h.get("name", "")
        ts_code = code_to_tscode(code)
        item: dict[str, Any] = {
            "code": code, "name": name, "ts_code": ts_code,
            "fin_roe": None, "fin_debt_ratio": None,
            "fin_grossprofit_margin": None, "fin_net_profit_yoy": None,
            "news_titles": [],
        }
        indicators = get_financial_indicators(
            ts_code,
            fields=[
                "ts_code",
                "end_date",
                "eps",
                "roe",
                "grossprofit_margin",
                "debt_to_assets",
                "netprofit_yoy",
                "or_yoy",
            ],
        )
        if indicators:
            row = indicators[0]
            item.update(
                fin_roe=_safe_pct(row.get("roe")),
                fin_debt_ratio=_safe_pct(row.get("debt_to_assets")),
                fin_grossprofit_margin=_safe_pct(row.get("grossprofit_margin")),
                fin_net_profit_yoy=_safe_pct(row.get("netprofit_yoy"), 10000),
            )

        news = fetch_stock_news(code, top_n=5)
        item["news_titles"] = [
            str(record.get("新闻标题", ""))
            for record in news
            if record.get("新闻标题")
        ][:5]

        raw.append(item)
    return {"raw_data": raw}


def _llm_assess_company(state: CompanyState) -> dict:
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
            parsed = chat_json("你是一个A股投研分析师。输出 JSON，不要 markdown 包裹。", prompt)
        except Exception:
            parsed = {}
            logger.warning("[LLM] Company Agent 失败: %s", item["code"], exc_info=True)

        assessments.append({
            **item,
            "risk_level": parsed.get("risk_level", "a") if isinstance(parsed, dict) else "a",
            "llm_risk_signals": parsed.get("risk_signals", []) if isinstance(parsed, dict) else [],
            "llm_positives": parsed.get("positives", []) if isinstance(parsed, dict) else [],
            "llm_summary": parsed.get("summary", "LLM 分析暂不可用") if isinstance(parsed, dict) else "LLM 分析暂不可用",
        })
    return {"company_assessments": assessments}


company_wf = StateGraph(CompanyState)
company_wf.add_node("collect_data", _collect_company_data)
company_wf.add_node("llm_assess", _llm_assess_company)
company_wf.set_entry_point("collect_data")
company_wf.add_edge("collect_data", "llm_assess")
company_wf.add_edge("llm_assess", END)
company_agent = company_wf.compile()


class RiskState(TypedDict):
    holdings: list[dict[str, Any]]
    st_codes: list[str]
    suspended_codes: list[str]
    risk_assessments: list[dict[str, Any]]
    error: str | None


def _check_st_status(state: RiskState) -> dict:
    codes = [h["code"] for h in state.get("holdings", []) if h.get("code")]
    return {"st_codes": get_st_codes(codes)}


def _check_suspension(state: RiskState) -> dict:
    codes = [h["code"] for h in state.get("holdings", []) if h.get("code")]
    return {"suspended_codes": get_suspended_codes(codes)}


def _assess_risks(state: RiskState) -> dict:
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


class StrategyState(TypedDict):
    market_sentiment: str
    holdings: list[dict[str, Any]]
    analysis: list[dict[str, Any]]
    strategy_result: dict[str, Any]
    error: str | None


def _generate_strategy(state: StrategyState) -> dict:
    sentiment = state.get("market_sentiment", "")[:200]
    analysis = state.get("analysis", [])
    holdings = state.get("holdings", [])
    lines = [f"市场情绪：{sentiment}", f"持仓 {len(holdings)} 只："]
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
    prompt = "\n".join(lines) + (
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
        result = chat_json("你是一个A股策略分析师，综合市场、公司、风险数据给出建议。", prompt)
    except Exception:
        logger.warning("[LLM] Strategy Agent 调用失败", exc_info=True)
        result = {
            "market_judgment": "LLM 暂不可用",
            "position_suggestions": "维持现有持仓",
            "risk_reminders": "注意控制仓位",
            "tomorrow_focus": "关注市场变化",
            "overall_strategy": "观望",
        }
    return {"strategy_result": result if isinstance(result, dict) else {}}


strategy_wf = StateGraph(StrategyState)
strategy_wf.add_node("generate_strategy", _generate_strategy)
strategy_wf.set_entry_point("generate_strategy")
strategy_wf.add_edge("generate_strategy", END)
strategy_agent = strategy_wf.compile()


class ReportState(TypedDict):
    summary: dict[str, Any]
    analysis: list[dict[str, Any]]
    risk_signals: list[str]
    market_sentiment: str
    strategy_result: dict[str, Any]
    report_text: str
    error: str | None


def _generate_report(state: ReportState) -> dict:
    summary = state.get("summary", {})
    analysis = state.get("analysis", [])
    signals = state.get("risk_signals", [])
    sentiment = state.get("market_sentiment", "")[:200]
    strategy = state.get("strategy_result", {})
    lines = [f"市场情绪：{sentiment}", f"持仓 {summary.get('total_stocks', 0)} 只："]
    for a in analysis:
        sigs = "、".join(a.get("risk_signals", [])[:2]) or "无"
        lines.append(
            f"- {a.get('code','')} {a.get('name','')} "
            f"[{a.get('risk_level','')}] ROE={a.get('fin_roe','?')}% "
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
        result = chat_json("你是一个A股投资日报编辑，生成专业简洁的日报。", prompt)
    except Exception:
        logger.warning("[LLM] Report Agent 失败", exc_info=True)
        result = {"full_report": "日报生成暂不可用", "report_title": ""}
    return {"report_text": result.get("full_report", "") if isinstance(result, dict) else "", "report_data": result}


report_wf = StateGraph(ReportState)
report_wf.add_node("generate_report", _generate_report)
report_wf.set_entry_point("generate_report")
report_wf.add_edge("generate_report", END)
report_agent = report_wf.compile()


def run_market_agent(state: PortfolioState) -> dict:
    r = market_agent.invoke({"holdings": state.get("holdings", [])})
    return {
        "market_indices": r.get("market_indices", {}),
        "limit_up_codes": r.get("limit_up_codes", []),
        "market_assessments": r.get("market_assessments", []),
        "sentiment_analysis": r.get("sentiment_analysis", ""),
    }


def run_company_agent(state: PortfolioState) -> dict:
    r = company_agent.invoke({"holdings": state.get("holdings", [])})
    return {"company_assessments": r.get("company_assessments", [])}


def run_risk_agent(state: PortfolioState) -> dict:
    r = risk_agent.invoke({"holdings": state.get("holdings", [])})
    return {
        "st_codes": r.get("st_codes", []),
        "suspended_codes": r.get("suspended_codes", []),
        "risk_assessments": r.get("risk_assessments", []),
    }


def merge_analysis(state: PortfolioState) -> dict:
    market = state.get("market_assessments")
    company = state.get("company_assessments")
    risk_list = state.get("risk_assessments")
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

        valid = ("aaa", "aa", "a", "bb", "cc")
        if code in st_set:
            merged["risk_level"] = "cc"
        elif code in susp_set:
            merged["risk_level"] = "bb"
        else:
            llm_level = ca.get("risk_level", "").lower()
            if llm_level in valid:
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
    r = strategy_agent.invoke({
        "market_sentiment": state.get("sentiment_analysis", ""),
        "holdings": state.get("holdings", []),
        "analysis": state.get("analysis", []),
    })
    return {"strategy_result": r.get("strategy_result", {})}


def run_report_agent(state: PortfolioState) -> dict:
    result = report_agent.invoke({
        "summary": state.get("summary", {}),
        "analysis": state.get("analysis", []),
        "risk_signals": state.get("risk_signals", []),
        "market_sentiment": state.get("sentiment_analysis", ""),
        "strategy_result": state.get("strategy_result", {}),
    })
    return {"report_text": result.get("report_text", "")}


def _router(state: PortfolioState) -> str:
    return "error_end" if state.get("error") else "continue"


wg = StateGraph(PortfolioState)
for _name in (
    "parse_csv", "enrich_holdings", "generate_summary",
    "run_market_agent", "run_company_agent", "run_risk_agent", "merge_analysis",
    "run_strategy_agent", "run_report_agent",
):
    wg.add_node(_name, globals()[_name])
wg.set_entry_point("parse_csv")
wg.add_conditional_edges("parse_csv", _router, {"continue": "enrich_holdings", "error_end": END})
wg.add_edge("enrich_holdings", "run_market_agent")
wg.add_edge("enrich_holdings", "run_company_agent")
wg.add_edge("enrich_holdings", "run_risk_agent")
wg.add_edge(["run_market_agent", "run_company_agent", "run_risk_agent"], "merge_analysis")
wg.add_edge("merge_analysis", "run_strategy_agent")
wg.add_conditional_edges("run_strategy_agent", _router, {"continue": "run_report_agent", "error_end": END})
wg.add_conditional_edges("run_report_agent", _router, {"continue": "generate_summary", "error_end": END})
wg.add_edge("generate_summary", END)
portfolio_agent = wg.compile()


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_pct(v: Any, max_val: float = 100) -> float | None:
    val = _safe_float(v)
    if val is None:
        return None
    if abs(val) > max_val * 10:
        return None
    return val
