"""Stock-level data-source helpers."""

from __future__ import annotations

from typing import Any

import akshare as ak
import pandas as pd
import tushare as ts

from src.logs import get_logger
from src.utils import tushare_api_key

logger = get_logger(__name__)

_pro = ts.pro_api(tushare_api_key())


def code_to_tscode(code: str) -> str:
    """Convert six-digit A-share code to Tushare ts_code."""
    code = str(code).strip().zfill(6)
    suffix = "SH" if code.startswith(("6", "9")) else "SZ"
    return f"{code}.{suffix}"


def get_stock_name_map() -> dict[str, str]:
    """Return mapping of stock code to stock name."""
    try:
        df = ak.stock_info_a_code_name()
        return dict(zip(df["code"].astype(str), df["name"].astype(str)))
    except Exception:
        logger.exception("Failed to fetch stock code/name map.")
        return {}


def get_st_codes(codes: list[str]) -> list[str]:
    """Return codes whose names indicate ST, *ST, or delisting risk."""
    target = {str(code).strip().zfill(6) for code in codes if code}
    if not target:
        return []
    try:
        df = ak.stock_info_a_code_name()
        st_codes: list[str] = []
        for _, row in df.iterrows():
            code = str(row.get("code", "")).zfill(6)
            name = str(row.get("name", ""))
            if code in target and ("ST" in name or "退" in name):
                st_codes.append(code)
        return st_codes
    except Exception:
        logger.exception("Failed to check ST stock names.")
        return []


def get_suspended_codes(codes: list[str]) -> list[str]:
    """Return codes that are currently fully suspended.

    Intraday suspensions with suspend_timing populated are ignored.
    """
    suspended: list[str] = []
    for code in [str(c).strip().zfill(6) for c in codes if c]:
        ts_code = code_to_tscode(code)
        try:
            df: pd.DataFrame = _pro.query("suspend_d", ts_code=ts_code)
            if df is None or df.empty:
                continue
            full = df[df["suspend_timing"].isna() | (df["suspend_timing"] == "")]
            if full.empty:
                continue
            latest = full.sort_values("trade_date", ascending=False).iloc[0]
            if latest.get("suspend_type") == "S":
                suspended.append(code)
        except Exception:
            logger.exception("Failed to check suspension for %s.", ts_code)
    return suspended
