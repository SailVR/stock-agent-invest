"""工具函数 — 环境变量访问器、安全类型转换等。"""

from __future__ import annotations

from typing import Any

from src.config import require_env, get_env


# ── API Key 访问器 ─────────────────────────────────────────────────────

def tushare_api_key() -> str:
    return require_env("TUSHARE_API")


def dashscope_api_key() -> str:
    return require_env("DASHSCOPE_API_KEY")


def deepseek_api_key() -> str:
    return require_env("DEEPSEEK_API_KEY")


def openai_api_key() -> str | None:
    return get_env("OPENAI_API_KEY")


# ── 安全类型转换 ───────────────────────────────────────────────────────

def safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def safe_pct(v: Any, max_val: float = 100) -> float | None:
    """安全的百分比转换，超合理范围则返回 None。"""
    val = safe_float(v)
    if val is None:
        return None
    if abs(val) > max_val * 10:
        return None
    return val


def safe_str(v: Any) -> str | None:
    if v is None:
        return None
    return str(v)
