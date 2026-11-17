"""Load config/settings.yaml with environment variable overrides.

Override any nested key with TABLETALK_<SECTION>__<KEY>, for example:
    TABLETALK_LLM__MAIN_MODEL=qwen3:4b
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "settings.yaml"
ENV_PREFIX = "TABLETALK_"


def _coerce(value: str) -> Any:
    """Turn env strings into numbers/bools where it makes sense."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = key[len(ENV_PREFIX):].lower().split("__")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return cfg


@lru_cache(maxsize=1)
def get_settings() -> dict[str, Any]:
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _apply_env_overrides(cfg)


def path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in settings."""
    return ROOT / get_settings()["paths"][key]
