#!/usr/bin/env python3
"""Flask entrypoint for the Portfolio Agent web app."""

from __future__ import annotations

import json
import os
import time

from flask import Flask, Response, render_template, request
from openai import OpenAI

from memory.long_term_memory import LongTermMemory
from memory.memory_extractor import MemoryExtractor
from memory.schema import init_memory_database
from src.agents import (
    build_stock_name_map,
    company_agent,
    market_agent,
    risk_agent,
    strategy_agent,
)
from src.config import llm_config
from src.logs import get_logger, setup_logging
from src.services.analysis_stream import run_analysis_stream, sse
from src.services.rag_provider import LazyRAGProvider
from src.services.stock_lookup import search_stock_code

logger = get_logger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(16).hex()

_result_cache: dict[str, dict] = {}
_RESULT_TTL = 300

memory = LongTermMemory()
extractor = MemoryExtractor()
rag_provider = LazyRAGProvider(os.path.dirname(__file__))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Stream CSV portfolio analysis progress."""
    if "file" not in request.files:
        return render_template("index.html", error="请选择文件")
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="文件名为空")
    if not file.filename.endswith(".csv"):
        return render_template("index.html", error="仅支持 CSV 文件")

    csv_text = file.read().decode("utf-8-sig")
    return Response(
        run_analysis_stream(csv_text, memory, rag_provider.get, _result_cache),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/result/<result_id>")
def show_result(result_id: str):
    """Display a cached analysis result."""
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


@app.route("/chat", methods=["POST"])
def chat_api():
    """Stream an LLM chat response with optional stock/portfolio context."""
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return {"error": "消息不能为空"}, 400

    def generate():
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

        intent_type = intent.get("type", "chat") if isinstance(intent, dict) else "chat"
        stock_code = None
        stock_name = ""
        if intent_type == "stock_query":
            stock_name = intent.get("stock_name") or user_msg
            stock_code = search_stock_code(stock_name)

        nl = "\n"
        context_parts = []

        yield f"data: {json.dumps({'token': '🔄 分析中'})}\n\n"

        if stock_code:
            query_name = intent.get("stock_name", stock_code) or stock_code
            yield f"data: {json.dumps({'token': f' 🔍 {query_name}'})}\n\n"

            name_map = build_stock_name_map()
            stock_name = name_map.get(stock_code, stock_code)

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

            yield f"data: {json.dumps({'token': ' 📈'})}\n\n"
            try:
                mr = market_agent.invoke({"holdings": [{"code": stock_code, "name": stock_name}]})
                idx = mr.get("market_indices", {})
                if idx:
                    parts = [f"{k} {v.get('pct_chg', '?')}%" for k, v in idx.items()]
                    context_parts.append(f"大盘表现：{' | '.join(parts)}")
                if stock_code in mr.get("limit_up_codes", []):
                    context_parts.append("📈 该股今日涨停")
            except Exception:
                logger.warning("[Chat] Market Agent 失败", exc_info=True)

            try:
                notes = memory.get_stock_notes(stock_code)
                if notes:
                    last = notes[0]
                    context_parts.append(
                        f"前次分析 ({last.get('created_at','?')})："
                        f"评级={last['risk_level']}, {last.get('analysis_summary','')}"
                    )
            except Exception:
                logger.warning("[Chat] 读取历史分析失败", exc_info=True)

            try:
                rag_results = rag_provider.get().search(f"{stock_code} {stock_name}", top_k=3)
                for item in rag_results:
                    if item.get("score", 0) > 0.3:
                        context_parts.append(
                            f"[历史] {item.get('company','')}: {item.get('content','')[:120]}"
                        )
            except Exception:
                logger.warning("[Chat] RAG 检索失败", exc_info=True)

        if intent_type == "strategy" and not stock_code:
            try:
                portfolios = memory.get_portfolio_history(limit=30)
                codes = list(dict.fromkeys(p["stock_code"] for p in portfolios if p.get("stock_code")))
                if codes:
                    yield f"data: {json.dumps({'token': f'📊 从历史记录加载 {len(codes)} 只持仓，开始分析...{nl}{nl}'})}\n\n"
                    name_map = build_stock_name_map()
                    holdings = [{"code": c, "name": name_map.get(c, c)} for c in codes]

                    yield sse("msg", {"token": "📈 拉取大盘数据 & 涨停池..."})
                    mr = market_agent.invoke({"holdings": holdings})
                    idx = mr.get("market_indices", {})
                    idx_str = "; ".join(f"{k} {v.get('pct_chg','?')}%" for k, v in idx.items())
                    limit_n = len(mr.get("limit_up_codes") or [])
                    yield sse("msg", {"token": f"✅ 大盘: {idx_str} | 涨停 {limit_n} 只"})

                    yield sse("msg", {"token": "📊 分析个股财务数据 & 新闻..."})
                    ca = company_agent.invoke({"holdings": holdings})
                    for item in (ca.get("company_assessments") or []):
                        yield sse("msg", {
                            "token": (
                                f"📊 {item.get('code','')} {item.get('name','')}: "
                                f"ROE={item.get('fin_roe','?')}% "
                                f"负债率={item.get('fin_debt_ratio','?')}% "
                                f"评级={item.get('risk_level','?')}"
                            )
                        })

                    yield sse("msg", {"token": "🔍 排查 ST / 停牌风险..."})
                    rr = risk_agent.invoke({"holdings": holdings})
                    for h in holdings:
                        code = h["code"]
                        risk_msgs = []
                        if code in (rr.get("st_codes") or []):
                            risk_msgs.append("ST")
                        if code in (rr.get("suspended_codes") or []):
                            risk_msgs.append("停牌")
                        if risk_msgs:
                            yield sse("msg", {"token": f"🔍 {code} {h.get('name','')}: {' '.join(risk_msgs)}"})
                    yield f"data: {json.dumps({'token': f'✅ 风险排查完成{nl}'})}\n\n"

                    st_set = set(rr.get("st_codes", []))
                    analysis = []
                    for h in holdings:
                        code = h["code"]
                        comp = next((x for x in (ca.get("company_assessments") or []) if x["code"] == code), {})
                        sigs = list(comp.get("llm_risk_signals", []))
                        if code in st_set:
                            sigs.append("ST/*ST")
                        analysis.append({
                            "code": code,
                            "name": h["name"],
                            "risk_level": comp.get("risk_level", "a"),
                            "risk_signals": sigs,
                            "fin_roe": comp.get("fin_roe"),
                            "fin_debt_ratio": comp.get("fin_debt_ratio"),
                            "fin_grossprofit_margin": comp.get("fin_grossprofit_margin"),
                            "fin_net_profit_yoy": comp.get("fin_net_profit_yoy"),
                        })

                    yield f"data: {json.dumps({'token': ' 🧠'})}\n\n"
                    sr = strategy_agent.invoke({
                        "market_sentiment": mr.get("sentiment_analysis", ""),
                        "holdings": holdings,
                        "analysis": analysis,
                    })
                    for key, value in (sr.get("strategy_result", {}) or {}).items():
                        if value:
                            context_parts.append(f"{key}: {value}")
            except Exception:
                logger.warning("[Chat] Strategy 分析失败", exc_info=True)

        try:
            interests = memory.get_interests(limit=5)
            if interests:
                ctx = extractor.format_interests_for_prompt(interests)
                if ctx:
                    context_parts.append(ctx)
        except Exception:
            logger.warning("[Chat] 读取用户偏好失败", exc_info=True)

        yield f"data: {json.dumps({'token': f' ✨{nl}{nl}'})}\n\n"

        system = "你是一个A股投研分析师。回答简洁专业，用中文。"
        if (stock_code or intent_type == "strategy") and context_parts:
            subject_name = stock_name if stock_code else "持仓组合"
            subject_code = stock_code or ""
            user_prompt = (
                f"用户问：{user_msg}\n\n"
                f"股票 {subject_name} ({subject_code}) 的相关数据：\n"
                + "\n".join(context_parts)
                + "\n\n请根据以上数据回答用户的问题。"
            )
        else:
            user_prompt = user_msg

        try:
            logger.info("[LLM] Chat API 用户问题: %s", user_msg[:200])
            config = llm_config()
            client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
            stream = client.chat.completions.create(
                model=config["model"],
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

            try:
                interests = extractor.extract_interests_from_conversation(user_msg, "".join(full_reply))
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
                logger.warning("[Memory] 提取用户偏好失败", exc_info=True)
        except Exception as e:
            yield f"data: {json.dumps({'token': f'❌ LLM 调用失败: {e}'})}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/memory/interests")
def api_memory_interests():
    try:
        items = memory.get_interests(limit=30)
        return {"items": items, "total": len(items)}
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/api/memory/history")
def api_memory_history():
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
    try:
        db_path = os.path.join(os.path.dirname(__file__), "data", "memory.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            init_memory_database(db_path)
            logger.info("[Memory] 数据库已清空")
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    setup_logging()
    app.run(debug=True, port=5001, threaded=True)
