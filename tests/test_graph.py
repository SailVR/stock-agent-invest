#!/usr/bin/env python3
"""输出所有 LangGraph Agent 的结构图。

用法：
    python tests/test_graph.py          # 控制台 ASCII 图
    python tests/test_graph.py --png    # 保存 PNG 图片（需安装 graphviz）
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 必须先 setup_logging 避免重复初始化
from src.logs import setup_logging
setup_logging()

# 导入所有 agent
from app import portfolio_agent, market_agent, company_agent, risk_agent, strategy_agent, report_agent


def show_graph(name: str, agent, save_png: bool = False):
    """打印 Agent 的图结构，并可选择保存 PNG。"""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    # 1. ASCII 图
    print("\n【结构图】")
    try:
        print(agent.get_graph().draw_ascii())
    except Exception as e:
        print(f"  (ASCII 渲染失败: {e})")

    # 2. 节点列表
    print("【节点】")
    try:
        for node_name, node_data in agent.get_graph().nodes.items():
            print(f"  ─ {node_name}")
    except Exception:
        pass

    # 3. PNG（可选，需 graphviz）
    if save_png:
        try:
            png = agent.get_graph().draw_mermaid_png()
            filename = f"{name.replace(' ', '_')}.png"
            with open(filename, "wb") as f:
                f.write(png)
            print(f"  ✅ 已保存: {filename}")
        except Exception as e:
            print(f"  ⚠️  PNG 导出失败（需安装 graphviz）: {e}")


def main():
    save_png = "--png" in sys.argv

    agents = [
        ("Portfolio Agent（主图）", portfolio_agent),
        ("Market Agent（子图）", market_agent),
        ("Company Agent（子图）", company_agent),
        ("Risk Agent（子图）", risk_agent),
        ("Strategy Agent（子图）", strategy_agent),
        ("Report Agent（子图）", report_agent),
    ]

    for name, agent in agents:
        show_graph(name, agent, save_png)


if __name__ == "__main__":
    main()
