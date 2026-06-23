#!/usr/bin/env python3
"""Quick verification script for Company Tools (Tushare financial data).

Run:  python tests/test_company_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tools.company_tools import (
    get_financial_indicators,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    fetch_stock_news,
)


def _sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_indicators() -> None:
    _sep("1. get_financial_indicators — 财务指标")
    # 贵州茅台
    data = get_financial_indicators("600519.SH")
    if not data:
        print("  ⚠  No data returned.")
        return
    r = data[0]
    print(f"  股票: {r['ts_code']}")
    print(f"  报告期: {r['end_date']}")
    print(f"  EPS: {r.get('eps', '?')}")
    print(f"  ROE: {r.get('roe', '?')}%")
    print(f"  毛利率: {r.get('gross_margin', '?')}%")
    print(f"  资产负债率: {r.get('debt_to_assets', '?')}%")
    print(f"  经营活动现金流/股: {r.get('ocfps', '?')}")
    print(f"  ROE(摊薄): {r.get('roe_dt', '?')}%")
    print(f"\n  ✅ 获取到 {len(data)} 期财务指标")


def test_income() -> None:
    _sep("2. get_income_statement — 利润表")
    data = get_income_statement("600519.SH")
    if not data:
        print("  ⚠  No data returned.")
        return
    r = data[0]
    print(f"  股票: {r['ts_code']}")
    print(f"  报告期: {r['end_date']}")
    print(f"  营业总收入: {r.get('revenue', '?')}")
    print(f"  营业利润: {r.get('operate_profit', '?')}")
    print(f"  净利润: {r.get('net_profit', '?')}")
    print(f"  EBITDA: {r.get('ebitda', '?')}")
    print(f"\n  ✅ 获取到 {len(data)} 期利润表")


def test_balance() -> None:
    _sep("3. get_balance_sheet — 资产负债表")
    data = get_balance_sheet("600519.SH")
    if not data:
        print("  ⚠  No data returned.")
        return
    r = data[0]
    print(f"  股票: {r['ts_code']}")
    print(f"  报告期: {r['end_date']}")
    print(f"  货币资金: {r.get('money_cap', '?')}")
    print(f"  总资产: {r.get('total_assets', '?')}")
    print(f"  总负债: {r.get('total_liab', '?')}")
    print(f"  股东权益: {r.get('total_hldr_eqy_exc_min_int', '?')}")
    print(f"\n  ✅ 获取到 {len(data)} 期资产负债表")


def test_cashflow() -> None:
    _sep("4. get_cash_flow — 现金流量表")
    data = get_cash_flow("600519.SH")
    if not data:
        print("  ⚠  No data returned.")
        return
    r = data[0]
    print(f"  股票: {r['ts_code']}")
    print(f"  报告期: {r['end_date']}")
    print(f"  经营活动现金流: {r.get('operate_cashflow', '?')}")
    print(f"  投资活动现金流: {r.get('invest_cashflow', '?')}")
    print(f"  筹资活动现金流: {r.get('fin_cashflow', '?')}")
    print(f"  自由现金流: {r.get('free_cashflow', '?')}")
    print(f"\n  ✅ 获取到 {len(data)} 期现金流量表")


def test_news() -> None:
    _sep("5. fetch_stock_news — 个股新闻")
    data = fetch_stock_news("002202", top_n=5)
    if not data:
        print("  ⚠  No data returned.")
        return
    for i, item in enumerate(data, 1):
        print(f"  #{i} [{item.get('发布时间', '?')}] {item.get('新闻标题', '?')}")
        print(f"      来源: {item.get('文章来源', '?')}")
    print(f"\n  ✅ 成功获取 {len(data)} 条新闻")


def main() -> None:
    print(">>> 开始验证 Company Tools (Tushare) <<<")

    tests = [
        ("get_financial_indicators", test_indicators),
        ("get_income_statement", test_income),
        ("get_balance_sheet", test_balance),
        ("get_cash_flow", test_cashflow),
        ("fetch_stock_news", test_news),
    ]

    errors = []
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
