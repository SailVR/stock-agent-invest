"""
文档元数据存储 — SQLite，记录每条向量的来源、公司、时间、内容等。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.logs import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    company    TEXT,
    sector     TEXT DEFAULT '',
    event_type TEXT DEFAULT 'analysis',
    time       TEXT,
    source     TEXT DEFAULT 'agent',
    content    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_docs_company ON documents(company);
CREATE INDEX IF NOT EXISTS idx_docs_event  ON documents(event_type);
CREATE INDEX IF NOT EXISTS idx_docs_time   ON documents(time);
"""


class DocumentStore:
    """文档元数据存储。"""

    def __init__(self, db_path: str | Path = "data/rag_docs.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, company: str, content: str, sector: str = "",
            event_type: str = "analysis", source: str = "agent",
            time: str | None = None) -> int:
        """添加一条文档记录，返回 doc_id。"""
        if time is None:
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO documents (company, sector, event_type, time, source, content) "
                "VALUES (?,?,?,?,?,?)",
                (company, sector, event_type, time, source, content),
            )
            conn.commit()
            return cur.lastrowid or 0

    def get(self, doc_id: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def search_by_company(self, company: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE company LIKE ? ORDER BY time DESC LIMIT ?",
                (f"%{company}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_by_event(self, event_type: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE event_type=? ORDER BY time DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
