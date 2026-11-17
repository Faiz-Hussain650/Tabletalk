"""Tiny SQLite store for confirmed orders (shared by the agent, MCP server and kitchen display)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from agent.order_state import Order

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    customer_name TEXT,
    fulfilment TEXT,
    total REAL NOT NULL,
    items_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'new'      -- new | preparing | ready | done
);
"""


class OrderDB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def save_order(self, order: Order) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO orders (created_at, customer_name, fulfilment, total, items_json) VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    order.customer_name,
                    order.fulfilment,
                    order.total,
                    json.dumps([i.model_dump() for i in order.items]),
                ),
            )
            return int(cur.lastrowid or 0)

    def list_orders(self, since_id: int = 0, limit: int = 50, include_done: bool = False) -> list[dict]:
        sql = "SELECT * FROM orders WHERE id > ?"
        if not include_done:
            sql += " AND status != 'done'"
        sql += " ORDER BY id ASC LIMIT ?"
        with self._conn() as c:
            rows = c.execute(sql, (since_id, limit)).fetchall()
        return [dict(r) | {"items": json.loads(r["items_json"])} for r in rows]

    def set_status(self, order_id: int, status: str) -> None:
        if status not in {"new", "preparing", "ready", "done"}:
            raise ValueError(f"bad status {status}")
        with self._conn() as c:
            c.execute("UPDATE orders SET status = ? WHERE id = ?", (status, order_id))
