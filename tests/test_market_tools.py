#!/usr/bin/env python3
"""Quick verification script for all Market Tools.

Run:  python tests/test_market_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `import src` works
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tools.market_tools import (
    get_market_index,
    get_north_flow,
    get_limit_up_stocks,
)


def _print_separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_market_index() -> None:
    _print_separator("1. get_market_index — 市场指数行情")
    result = get_market_index()
    if not result:
        print("  ⚠  No data returned (possibly non-trading day).")
        return

    for name, data in result.items():
        print(f"  {name} ({data['ts_code']})")
        print(f"    日期: {data['trade_date']}")
        print(f"    开盘: {data['open']}")
        print(f"    收盘: {data['close']}")
        print(f"    最高: {data['high']}")
        print(f"    最低: {data['low']}")
        print(f"    涨幅: {data['pct_chg']}%")
    print(f"\n  ✅ 成功获取 {len(result)} 个指数数据")



def test_north_flow() -> None:
    _print_separator("2. get_north_flow — 北向资金")
    data = get_north_flow(days=3)
    if not data:
        print("  ⚠  No data returned.")
        return
    for row in data:
        print(f"  {row['trade_date']}  北向: {row.get('north_money', '?')}  沪股通: {row.get('ggt_ss', '?')}  深股通: {row.get('ggt_sz', '?')}")
    print(f"\n  ✅ 成功获取 {len(data)} 天北向资金数据")


def test_limit_up_stocks() -> None:
    _print_separator("2. get_limit_up_stocks — 涨停股票")
    result = get_limit_up_stocks()
    if not result:
        print("  ⚠  No limit-up stocks today (or non-trading day).")
        return

    for i, stock in enumerate(result[:5], 1):
        code = stock.get("代码", "?")
        name = stock.get("名称", "?")
        price = stock.get("最新价", "?")
        print(f"  #{i}  {code} {name}  |  最新价: {price}")
    if len(result) > 5:
        print(f"  ... 共 {len(result)} 只涨停股票（仅展示前5只）")
    print(f"\n  ✅ 成功获取涨停板数据（共 {len(result)} 只）")


def main() -> None:
    print(">>> 开始验证 Market Tools <<<")

    errors = []
    tests = [
        ("get_market_index", test_market_index),
        ("get_north_flow", test_north_flow),
        ("get_limit_up_stocks", test_limit_up_stocks),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n  ❌ {name} 执行失败: {e}")
            errors.append(name)

    print(f"\n{'=' * 60}")
    if errors:
        print(f"  部分失败: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("  全部验证通过 ✅")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
