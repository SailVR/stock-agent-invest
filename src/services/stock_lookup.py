"""Stock name/code lookup helpers."""

from __future__ import annotations

import akshare as ak

_stock_name_cache: dict[str, str] | None = None


def build_name_code_map() -> dict[str, str]:
    global _stock_name_cache
    if _stock_name_cache is not None:
        return _stock_name_cache
    try:
        df = ak.stock_info_a_code_name()
        _stock_name_cache = dict(zip(df["name"], df["code"]))
        return _stock_name_cache
    except Exception:
        return {}


def search_stock_code(query: str) -> str | None:
    mapping = build_name_code_map()
    if query in mapping:
        return mapping[query]
    for name, code in mapping.items():
        if query in name or name in query:
            return code
    return None
