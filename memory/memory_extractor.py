"""
记忆提取器 — 从对话中自动提取用户对股票的关注偏好。
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.llm import chat_json
from src.logs import get_logger

logger = get_logger(__name__)


class MemoryExtractor:
    """从对话中提取用户对股票的关注偏好。"""

    def extract_interests_from_conversation(
        self,
        user_msg: str,
        reply: str,
    ) -> list[dict[str, Any]]:
        """从单轮对话中提取用户关注偏好。

        Args:
            user_msg: 用户本轮消息。
            reply: 助手本轮回复。

        Returns:
            [{category, content, confidence}, ...]
        """
        prompt = (
            f"用户说：{user_msg}\n"
            f"助手回复：{reply}\n\n"
            f"分析以上对话，提取用户对A股投资的关注偏好。\n"
            f"类别包括：关注行业、投资风格、风险偏好、常用指标、具体股票。\n\n"
            f"输出 JSON 数组：\n"
            f'[{{"category":"关注行业","content":"新能源","confidence":0.9}},...]\n'
            f"如果没有值得记录的偏好，返回 []。"
        )
        try:
            result = chat_json(
                "你是一个用户偏好分析师，从对话中提取股票投资相关的关注点。",
                prompt,
            )
            if isinstance(result, list):
                valid = []
                for item in result:
                    if item.get("category") and item.get("content"):
                        valid.append(item)
                return valid
            return []
        except Exception:
            logger.warning("[MemoryExtractor] LLM 提取偏好失败", exc_info=True)
            return []

    @staticmethod
    def format_interests_for_prompt(interests: list[dict[str, Any]]) -> str:
        """将偏好列表格式化为 LLM 上下文文本。"""
        if not interests:
            return ""
        lines = ["用户近期关注偏好："]
        for item in interests:
            lines.append(f"- [{item.get('category','?')}] {item.get('content','')}")
        return "\n".join(lines)
