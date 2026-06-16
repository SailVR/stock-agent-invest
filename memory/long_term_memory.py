"""
长期记忆管理器 — 投研场景：持仓快照、个股分析记录、用户关注偏好。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from memory.schema import init_memory_database
from src.logs import get_logger

logger = get_logger(__name__)

DEFAULT_USER = "default_user"


class LongTermMemory:
    """长期记忆管理器 — 跨会话持久化投研信息。"""

    def __init__(self, db_path: str | Path = "data/memory.db"):
        self.db_path = str(db_path)
        if not Path(self.db_path).exists():
            init_memory_database(self.db_path)
        else:
            # ensure tables exist
            init_memory_database(self.db_path)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── User ────────────────────────────────────────────────────────

    def ensure_user(self, user_id: str = DEFAULT_USER) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO users (user_id, created_at, last_active)
                   VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                   ON CONFLICT(user_id) DO UPDATE SET last_active=CURRENT_TIMESTAMP""",
                (user_id,),
            )
            conn.commit()

    # ── Portfolio snapshots ─────────────────────────────────────────

    def save_portfolio(
        self,
        holdings: list[dict[str, Any]],
        user_id: str = DEFAULT_USER,
    ) -> None:
        """保存一次持仓快照。"""
        self.ensure_user(user_id)
        today = datetime.now().strftime("%Y-%m-%d")
        with self._conn() as conn:
            for h in holdings:
                conn.execute(
                    """INSERT INTO portfolios
                       (user_id, upload_date, stock_code, stock_name, prey)
                       VALUES (?, ?, ?, ?, ?)""",
                    (user_id, today, h.get("code"), h.get("name"), h.get("prey")),
                )
            conn.commit()
        logger.info("[Memory] 保存持仓快照 %d 条 → %s", len(holdings), today)

    def get_portfolio_history(
        self,
        user_id: str = DEFAULT_USER,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """获取最近 N 次持仓记录。"""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT upload_date, stock_code, stock_name, prey
                   FROM portfolios WHERE user_id=?
                   ORDER BY id DESC LIMIT ?""",
                (user_id, limit * 10),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_portfolio_by_date(
        self,
        date: str,
        user_id: str = DEFAULT_USER,
    ) -> list[dict[str, Any]]:
        """获取指定日期的持仓列表。"""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT stock_code, stock_name, prey
                   FROM portfolios WHERE user_id=? AND upload_date=?""",
                (user_id, date),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Stock notes ─────────────────────────────────────────────────

    def save_stock_note(
        self,
        code: str,
        name: str,
        risk_level: str,
        summary: str,
        fin_roe: float | None = None,
        fin_debt_ratio: float | None = None,
        fin_grossprofit_margin: float | None = None,
        fin_net_profit_yoy: float | None = None,
        market_sentiment: str = "",
        user_id: str = DEFAULT_USER,
    ) -> None:
        """保存一次个股分析结论。"""
        self.ensure_user(user_id)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO stock_notes
                   (user_id, stock_code, stock_name, risk_level,
                    analysis_summary, fin_roe, fin_debt_ratio,
                    fin_grossprofit_margin, fin_net_profit_yoy,
                    market_sentiment)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (user_id, code, name, risk_level, summary,
                 fin_roe, fin_debt_ratio, fin_grossprofit_margin,
                 fin_net_profit_yoy, market_sentiment),
            )
            conn.commit()
        logger.info("[Memory] 保存分析记录 %s %s → %s", code, name, risk_level)

    def get_stock_notes(
        self,
        code: str,
        user_id: str = DEFAULT_USER,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """获取某只股票的历史分析记录。"""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT stock_code, stock_name, risk_level, analysis_summary,
                          fin_roe, fin_debt_ratio, market_sentiment, created_at
                   FROM stock_notes
                   WHERE user_id=? AND stock_code=?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, code, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── User interests ──────────────────────────────────────────────

    def save_interest(
        self,
        category: str,
        content: str,
        confidence: float = 0.8,
        source: str = "chat",
        user_id: str = DEFAULT_USER,
    ) -> None:
        """保存一条用户关注偏好。"""
        self.ensure_user(user_id)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO user_interests
                   (user_id, category, content, confidence, source)
                   VALUES (?,?,?,?,?)""",
                (user_id, category, content, confidence, source),
            )
            conn.commit()

    def get_interests(
        self,
        category: str | None = None,
        user_id: str = DEFAULT_USER,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """获取用户关注偏好。"""
        with self._conn() as conn:
            if category:
                rows = conn.execute(
                    """SELECT category, content, confidence, created_at
                       FROM user_interests
                       WHERE user_id=? AND category=?
                       ORDER BY confidence DESC, created_at DESC LIMIT ?""",
                    (user_id, category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT category, content, confidence, created_at
                       FROM user_interests
                       WHERE user_id=?
                       ORDER BY created_at DESC LIMIT ?""",
                    (user_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]
