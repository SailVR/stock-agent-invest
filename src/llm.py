"""
LLM 封装 — 通过 OpenAI 兼容接口调用模型（支持 DashScope / DeepSeek）。
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from src.config import llm_config

_config = llm_config()
_client = OpenAI(
    api_key=_config["api_key"],
    base_url=_config["base_url"],
)


def chat(
    system: str,
    user: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """调用 LLM，返回文本回复。"""
    resp = _client.chat.completions.create(
        model=_config["model"],
        temperature=temperature or _config["temperature"],
        max_tokens=max_tokens or _config["max_tokens"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def chat_json(
    system: str,
    user: str,
    temperature: float | None = None,
) -> dict[str, Any]:
    """调用 LLM，要求输出 JSON，解析返回。

    DeepSeek 不支持 response_format，所以用纯文本方式 + 正则提取 JSON。
    """
    resp = _client.chat.completions.create(
        model=_config["model"],
        temperature=temperature or _config["temperature"],
        max_tokens=_config["max_tokens"],
        messages=[
            {"role": "system", "content": system + "\n\n请只输出 JSON，不要包含其他文字，不要用 markdown 包裹。"},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content or "{}"

    # 1. 找 ```json ... ``` 或 ``` ... ```
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        text = m.group(1).strip()
    # 2. 找最外层 { ... }（非贪婪，取第一个完整 JSON）
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        text = text[brace_start:brace_end + 1]
    # 3. 解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 再尝试：找到第一个完整 JSON 对象（处理多个 {} 的情况）
        depth = 0
        for i, c in enumerate(text):
            if c == "{":
                if depth == 0:
                    brace_start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    text = text[brace_start:i + 1]
                    break
        return json.loads(text)
