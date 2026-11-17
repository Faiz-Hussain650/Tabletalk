"""Tracing: always writes JSONL to traces/, and also sends to Langfuse if it is configured.

Langfuse is optional. Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST
(e.g. http://localhost:3000 for a self-hosted instance) to turn it on.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, folder: str | Path, session_id: str):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.file = self.folder / f"{time.strftime('%Y%m%d')}.jsonl"
        self._langfuse = None
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import get_client

                self._langfuse = get_client()
            except Exception:  # noqa: BLE001 - tracing must never break the agent
                self._langfuse = None

    def log(self, name: str, input: Any = None, output: Any = None, **metadata: Any) -> None:
        event = {
            "ts": time.time(),
            "session_id": self.session_id,
            "name": name,
            "input": input,
            "output": output,
            "metadata": metadata,
        }
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        if self._langfuse is not None:
            try:
                with self._langfuse.start_as_current_span(
                    name=name, input=input, metadata={"session_id": self.session_id, **metadata}
                ) as span:
                    span.update(output=output)
            except Exception:  # noqa: BLE001
                pass

    def flush(self) -> None:
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception:  # noqa: BLE001
                pass
