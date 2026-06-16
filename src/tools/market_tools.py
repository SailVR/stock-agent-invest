"""
Market Tools — data-source layer for market-wide information.

All functions return dict / list-of-dicts (JSON-serialisable) so they can be
consumed directly by agents without DataFrame dependency.
"""

from __future__ import annotations

from datetime import date as date_type
from typing import Any

import akshare as ak
import pandas as pd
import tushare as ts

from src.logs import get_logger
from src.utils import tushare_api_key

logger = get_logger(__name__)

# ── Tushare Pro ─────────────────────────────────────────────────────────
_pro = ts.pro_api(
    tushare_api_key()
)


# ──────────────────────────────────────────────────────────────────────
# 1. 市场指数行情
# ──────────────────────────────────────────────────────────────────────

MAJOR_INDICES: dict[str, str] = {
    "上证指数": "000001.SH",
    "深证成指": "399001.SZ",
    "创业板指": "399006.SZ",
    "科创50": "000688.SH",
    "沪深300": "000300.SH",
}


def get_market_index(
    symbols: dict[str, str] | None = None,
    trade_date: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch latest daily OHLCV for major Chinese stock indices via Tushare.

    Args:
        symbols: Dict of ``{name: tushare_ts_code}``.  Defaults to MAJOR_INDICES.
        trade_date: Date string ``"YYYYMMDD"``.  Defaults to latest trading day.

    Returns:
        Nested dict — ``{index_name: {ts_code, name, date, open, close, …}}``.
    """
    symbols = symbols or MAJOR_INDICES
    result: dict[str, dict[str, Any]] = {}

    for name, ts_code in symbols.items():
        try:
            df: pd.DataFrame = _pro.index_daily(
                ts_code=ts_code,
                start_date=trade_date,
                end_date=trade_date,
            )
            if df is None or df.empty:
                # fallback: fetch last 5 days and take the latest
                df = _pro.index_daily(ts_code=ts_code, start_date="", end_date="")
                if df is None or df.empty:
                    logger.warning("No data for index %s (%s)", name, ts_code)
                    continue

            latest = df.iloc[0]  # newest first from Tushare
            row = {
                "ts_code": ts_code,
                "name": name,
                "trade_date": str(latest.get("trade_date", "")),
                "open": _safe_float(latest.get("open")),
                "close": _safe_float(latest.get("close")),
                "high": _safe_float(latest.get("high")),
                "low": _safe_float(latest.get("low")),
                "pre_close": _safe_float(latest.get("pre_close")),
                "change": _safe_float(latest.get("change")),
                "pct_chg": _safe_float(latest.get("pct_chg")),
                "vol": _safe_float(latest.get("vol")),
                "amount": _safe_float(latest.get("amount")),
            }
            result[name] = row
            logger.info("Fetched index: %s @ %.2f", name, row["close"])

        except Exception:
            logger.exception("Failed to fetch index %s (%s)", name, ts_code)

    return result


# ──────────────────────────────────────────────────────────────────────
# 2. 北向资金
# ──────────────────────────────────────────────────────────────────────


def get_north_flow(days: int = 5) -> list[dict[str, Any]]:
    """Fetch north-bound capital flow (北向资金) via Tushare.

    Args:
        days: Number of recent trading days.

    Returns:
        List of daily north-flow records — ``[{trade_date, north_money,
        ggt_ss, ggt_sz, hgt, sgt, south_money}]``.
    """
    try:
        df: pd.DataFrame = _pro.moneyflow_hsgt(
            start_date="",
            end_date="",
        )
        if df is None or df.empty:
            logger.warning("No north-flow data returned.")
            return []

        df = df.head(days)
        records = df.to_dict(orient="records")
        _sanitise(records)
        logger.info("Fetched %d days of north-bound flow.", len(records))
        return records

    except Exception:
        logger.exception("Failed to fetch north flow.")
        return []


# ──────────────────────────────────────────────────────────────────────
# 3. 涨停股票
# ──────────────────────────────────────────────────────────────────────
# NOTE: Tushare 没有涨停板股票池接口，保留 AKShare (push2ex 通道，速度正常)


def get_limit_up_stocks(
    trade_date: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch today's limit-up (涨停) stocks from East Money via AKShare.

    Args:
        trade_date: Date string ``"YYYYMMDD"``.  Defaults to today.

    Returns:
        List of limit-up stock dicts — ``[{代码, 名称, 最新价, 涨停封单, 封成比, …}]``.
    """
    if trade_date is None:
        trade_date = date_type.today().strftime("%Y%m%d")

    try:
        df: pd.DataFrame = ak.stock_zt_pool_em(date=trade_date)
        if df.empty:
            logger.info("No limit-up stocks on %s.", trade_date)
            return []

        records = df.to_dict(orient="records")
        _sanitise(records)
        logger.info(
            "Fetched %d limit-up stocks on %s.",
            len(records),
            trade_date,
        )
        return records

    except Exception:
        logger.exception(
            "Failed to fetch limit-up stocks for %s.", trade_date,
        )
        return []


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_str(v: Any) -> str | None:
    if v is None:
        return None
    return str(v)


def _sanitise(records: list[dict]) -> None:
    """Convert numpy / Timestamp scalars to Python native types."""
    for r in records:
        for k, v in r.items():
            if hasattr(v, "item"):
                r[k] = v.item()
            elif isinstance(v, pd.Timestamp):
                r[k] = str(v)
