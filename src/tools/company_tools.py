"""
Company Tools — financial data sourced from Tushare Pro.

All functions return dict / list-of-dicts (JSON-serialisable) for agent
consumption without DataFrame dependency.
"""

from __future__ import annotations

import time
from typing import Any

import akshare as ak
import pandas as pd
import tushare as ts

from src.logs import get_logger
from src.utils import tushare_api_key

logger = get_logger(__name__)

# ── Tushare Pro initialisation (hardcoded token) ───────────────────────
_pro = ts.pro_api(
    tushare_api_key()
)


# ──────────────────────────────────────────────────────────────────────
# 1. 财务指标
# ──────────────────────────────────────────────────────────────────────

_FINA_INDICATOR_FIELDS = [
    "ts_code", "end_date", "eps", "dt_eps",
    "bps", "ocfps", "cfps",
    "roe", "roe_waa", "roe_dt", "roa", "roic",
    "gross_margin", "netprofit_margin", "grossprofit_margin",
    "debt_to_assets", "assets_to_eqt",
    "current_ratio", "quick_ratio",
    "op_income", "ebit", "ebitda",
    "fcff", "fcfe",
    "profit_dedt",
    "basic_eps_yoy", "roe_yoy",
    "tr_yoy", "or_yoy", "netprofit_yoy",
    "equity_yoy",
]


def get_financial_indicators(
    ts_code: str,
    start_date: str = "",
    end_date: str = "",
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch key financial indicators (财务指标) for a stock.

    Args:
        ts_code: Tushare code, e.g. ``"600519.SH"``.
        start_date: Start date ``"YYYYMMDD"`` (optional).
        end_date: End date ``"YYYYMMDD"`` (optional).
        fields: Subset of columns to return; defaults to *FINA_INDICATOR_FIELDS*.

    Returns:
        List of indicator dicts, newest period first.
    """
    if fields is None:
        fields = _FINA_INDICATOR_FIELDS

    try:
        df: pd.DataFrame = _pro.query(
            "fina_indicator",
            ts_code=ts_code,
            start_date=start_date or None,
            end_date=end_date or None,
            fields=",".join(fields),
        )
        if df is None or df.empty:
            logger.warning("No fina_indicator for %s.", ts_code)
            return []

        df = df.sort_values("end_date", ascending=False)
        records = df.to_dict(orient="records")
        _sanitise(records)
        logger.info(
            "Fetched %d indicator periods for %s.",
            len(records),
            ts_code,
        )
        return records

    except Exception:
        logger.exception("Failed to fetch fina_indicator for %s.", ts_code)
        return []


# ──────────────────────────────────────────────────────────────────────
# 2. 利润表 (Income Statement)
# ──────────────────────────────────────────────────────────────────────

_INCOME_FIELDS = [
    "ts_code", "end_date", "report_type",
    "revenue", "total_revenue",
    "operate_cost", "operate_expense",
    "sale_expense", "admin_expense", "fin_expense",
    "operate_profit", "total_profit",
    "income_net", "minority_interest",
    "net_profit", "fcff", "ebit", "ebitda",
]


def get_income_statement(
    ts_code: str,
    start_date: str = "",
    end_date: str = "",
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch income statement (利润表) for a stock.

    Args:
        ts_code: Tushare code, e.g. ``"600519.SH"``.
        start_date: Start date ``"YYYYMMDD"``.
        end_date: End date ``"YYYYMMDD"``.
        fields: Subset of columns; defaults to *_INCOME_FIELDS*.

    Returns:
        List of income-statement dicts, newest period first.
    """
    if fields is None:
        fields = _INCOME_FIELDS

    try:
        df: pd.DataFrame = _pro.query(
            "income",
            ts_code=ts_code,
            start_date=start_date or None,
            end_date=end_date or None,
            fields=",".join(fields),
        )
        if df is None or df.empty:
            logger.warning("No income statement for %s.", ts_code)
            return []

        df = df.sort_values("end_date", ascending=False)
        records = df.to_dict(orient="records")
        _sanitise(records)
        logger.info(
            "Fetched %d income-statement periods for %s.",
            len(records),
            ts_code,
        )
        return records

    except Exception:
        logger.exception("Failed to fetch income for %s.", ts_code)
        return []


# ──────────────────────────────────────────────────────────────────────
# 3. 资产负债表 (Balance Sheet)
# ──────────────────────────────────────────────────────────────────────

_BALANCE_FIELDS = [
    "ts_code", "end_date", "report_type",
    "total_share",
    "money_cap",
    "trad_asset",
    "notes_receiv",
    "accounts_receiv",
    "total_current_assets",
    "fixed_assets",
    "intan_assets",
    "total_assets",
    "accounts_pay",
    "total_current_liab",
    "total_liab",
    "total_hldr_eqy_exc_min_int",
    "minority_int",
]


def get_balance_sheet(
    ts_code: str,
    start_date: str = "",
    end_date: str = "",
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch balance sheet (资产负债表) for a stock.

    Args:
        ts_code: Tushare code, e.g. ``"600519.SH"``.
        start_date: Start date ``"YYYYMMDD"``.
        end_date: End date ``"YYYYMMDD"``.
        fields: Subset of columns; defaults to *_BALANCE_FIELDS*.

    Returns:
        List of balance-sheet dicts, newest period first.
    """
    if fields is None:
        fields = _BALANCE_FIELDS

    try:
        df: pd.DataFrame = _pro.query(
            "balancesheet",
            ts_code=ts_code,
            start_date=start_date or None,
            end_date=end_date or None,
            fields=",".join(fields),
        )
        if df is None or df.empty:
            logger.warning("No balance sheet for %s.", ts_code)
            return []

        df = df.sort_values("end_date", ascending=False)
        records = df.to_dict(orient="records")
        _sanitise(records)
        logger.info(
            "Fetched %d balance-sheet periods for %s.",
            len(records),
            ts_code,
        )
        return records

    except Exception:
        logger.exception(
            "Failed to fetch balance sheet for %s.", ts_code,
        )
        return []


# ──────────────────────────────────────────────────────────────────────
# 4. 现金流量表 (Cash Flow)
# ──────────────────────────────────────────────────────────────────────

_CASHFLOW_FIELDS = [
    "ts_code", "end_date", "report_type",
    "n_income",
    "n_income_attr_p",
    "operate_cashflow",
    "invest_cashflow",
    "fin_cashflow",
    "cash_eq_end",
    "free_cashflow",
]


def get_cash_flow(
    ts_code: str,
    start_date: str = "",
    end_date: str = "",
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch cash-flow statement (现金流量表) for a stock.

    Args:
        ts_code: Tushare code, e.g. ``"600519.SH"``.
        start_date: Start date ``"YYYYMMDD"``.
        end_date: End date ``"YYYYMMDD"``.
        fields: Subset of columns; defaults to *_CASHFLOW_FIELDS*.

    Returns:
        List of cash-flow dicts, newest period first.
    """
    if fields is None:
        fields = _CASHFLOW_FIELDS

    try:
        df: pd.DataFrame = _pro.query(
            "cashflow",
            ts_code=ts_code,
            start_date=start_date or None,
            end_date=end_date or None,
            fields=",".join(fields),
        )
        if df is None or df.empty:
            logger.warning("No cash-flow data for %s.", ts_code)
            return []

        df = df.sort_values("end_date", ascending=False)
        records = df.to_dict(orient="records")
        _sanitise(records)
        logger.info(
            "Fetched %d cash-flow periods for %s.",
            len(records),
            ts_code,
        )
        return records

    except Exception:
        logger.exception("Failed to fetch cash flow for %s.", ts_code)
        return []


# ──────────────────────────────────────────────────────────────────────
# 5. 个股新闻
# ──────────────────────────────────────────────────────────────────────


def fetch_stock_news(
    stock_code: str,
    delay: float = 0.6,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Fetch latest news for a stock from East Money via AKShare.

    Args:
        stock_code: Stock symbol, e.g. ``"002202"``.
        delay: Seconds to pause after fetch (rate-limit polite).
        top_n: Max number of news items to return.

    Returns:
        List of news dicts — ``[{发布时间, 新闻标题, 新闻内容, 文章来源, 新闻链接}]``.
    """
    try:
        df: pd.DataFrame = ak.stock_news_em(symbol=stock_code)
        time.sleep(delay)

        if df.empty:
            logger.warning("No news for stock %s.", stock_code)
            return []

        # Guard against missing columns
        expected = ["发布时间", "新闻标题", "新闻内容", "文章来源", "新闻链接"]
        have = [c for c in expected if c in df.columns]
        if not have:
            logger.warning("No expected columns in news response for %s.", stock_code)
            return []

        df = df.head(top_n)
        records = df[have].to_dict(orient="records")

        logger.info(
            "Fetched %d news items for stock %s.",
            len(records),
            stock_code,
        )
        return records

    except Exception:
        logger.exception("Failed to fetch news for stock %s.", stock_code)
        return []


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _sanitise(records: list[dict]) -> None:
    """Convert numpy scalars to Python native types in-place."""
    for r in records:
        for k, v in r.items():
            if hasattr(v, "item"):
                r[k] = v.item()
            elif isinstance(v, pd.Timestamp):
                r[k] = str(v)
