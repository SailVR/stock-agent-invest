#!/usr/bin/env python3
"""验证千问 LLM 是否可通过 DashScope API 正常调用。

用法：
    python tests/test_llm.py

期望输出：
    模型回复一段中文自我介绍。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm import chat, chat_json


def test_basic_chat() -> None:
    """测试基础对话。"""
    print("=" * 50)
    print("1. 基础对话测试")
    print("=" * 50)
    resp = chat(
        system="你是一个有用的助手。",
        user="请用一句话介绍你自己。",
    )
    print(f"  回复：{resp}")
    assert resp and len(resp) > 5, "回复过短或为空"
    print("  ✅ 基础对话通过\n")


def test_stock_analysis() -> None:
    """测试股票分析场景。"""
    print("=" * 50)
    print("2. 股票分析场景测试")
    print("=" * 50)
    resp = chat(
        system="你是一个A股投研分析师。",
        user=(
            "股票 600519.SH 贵州茅台\n"
            "财务指标：ROE=32.5%, 负债率=18.2%, 毛利率=91.8%, 净利同比+15.3%\n"
            "近期新闻标题：茅台提价预期升温；北向资金增持茅台\n\n"
            "请分析这只股票的基本面风险，给出 risk_level (safe/caution/danger) 和简短理由。"
        ),
    )
    print(f"  回复：{resp}")
    assert resp and len(resp) > 5, "回复过短或为空"
    print("  ✅ 股票分析通过\n")


def test_json_output() -> None:
    """测试 JSON 结构化输出。"""
    print("=" * 50)
    print("3. JSON 结构化输出测试")
    print("=" * 50)
    result = chat_json(
        system="你是一个金融数据提取助手。",
        user=(
            "上证指数今日涨跌幅 +1.2%，成交额 4500亿；"
            "深证成指涨跌幅 +2.1%，成交额 5800亿。"
            "请提取关键指标返回 JSON。"
        ),
    )
    print(f"  回复：{result}")
    assert isinstance(result, dict), "结果不是 dict"
    print("  ✅ JSON 输出通过\n")


def main() -> None:
    print(">>> 开始验证千问 LLM <<<\n")

    tests = [
        ("基础对话", test_basic_chat),
        ("股票分析", test_stock_analysis),
        ("JSON 输出", test_json_output),
    ]

    errors = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n  ❌ {name} 失败: {e}")
            errors.append(name)

    print("=" * 50)
    if errors:
        print(f"  部分失败: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("  全部验证通过 ✅")
        print("=" * 50)


if __name__ == "__main__":
    main()
