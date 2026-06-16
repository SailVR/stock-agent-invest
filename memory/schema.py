"""
Memory 数据库初始化 — 投研场景专用表结构。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def init_memory_database(db_path: str | Path) -> None:
    """初始化长期记忆数据库。

    Args:
        db_path: SQLite 文件路径。
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 用户表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id   TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 持仓快照表 — 每次上传 CSV 的记录
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            stock_code  TEXT NOT NULL,
            stock_name  TEXT,
            prey       REAL,
            risk_level TEXT,
            summary    TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_portfolios_user_code
        ON portfolios(user_id, stock_code)
    """)

    # 个股分析记录表 — Company Agent 每次分析的结果
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_notes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            stock_code  TEXT NOT NULL,
            stock_name  TEXT,
            risk_level  TEXT,
            analysis_summary TEXT,
            fin_roe     REAL,
            fin_debt_ratio REAL,
            fin_grossprofit_margin REAL,
            fin_net_profit_yoy REAL,
            market_sentiment TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_stock_notes_code
        ON stock_notes(user_id, stock_code, created_at)
    """)

    # 用户关注偏好表 — MemoryExtractor 从对话中提取
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_interests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            category    TEXT NOT NULL,
            content     TEXT NOT NULL,
            confidence  REAL DEFAULT 0.8,
            source      TEXT DEFAULT 'chat',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_interests_user
        ON user_interests(user_id, category)
    """)

    conn.commit()
    conn.close()
