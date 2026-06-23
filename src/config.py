"""Configuration — .env loader + 模型统一配置。"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── .env ───────────────────────────────────────────────────────────────
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


def get_env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def require_env(key: str) -> str:
    val = os.getenv(key)
    if val is None:
        raise RuntimeError(f"缺少环境变量: {key}")
    return val


# ── 项目路径 ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"

# ── LLM 配置 ───────────────────────────────────────────────────────────
# 切换 provider: "dashscope" 或 "deepseek"

LLM_PROVIDER = "dashscope"

# DashScope / 千问
DASHSCOPE_MODEL = "qwen-turbo"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# DeepSeek
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# 通用参数
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048


def llm_config() -> dict:
    """根据 LLM_PROVIDER 返回对应配置。"""
    if LLM_PROVIDER == "deepseek":
        return {
            "api_key": require_env("DEEPSEEK_API_KEY"),
            "base_url": DEEPSEEK_BASE_URL,
            "model": DEEPSEEK_MODEL,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        }
    # 默认 dashscope
    return {
        "api_key": require_env("DASHSCOPE_API_KEY"),
        "base_url": DASHSCOPE_BASE_URL,
        "model": DASHSCOPE_MODEL,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }


# ── Embedding / RAG ───────────────────────────────────────────────────

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# EmbeddingModel 会在加载后自动探测真实维度；该值只作为旧测试/临时向量用例的默认值。
EMBEDDING_DIM = 512
EMBEDDING_DEVICE = "cpu"
HF_ENDPOINT = "https://hf-mirror.com"
