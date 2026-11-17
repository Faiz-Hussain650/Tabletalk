"""Scoring functions shared by the text evals, voice evals and fine-tuning comparison."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from typing import Any

# ---------------------------------------------------------------- orders


def _unit_key(item: dict[str, Any]) -> tuple:
    return (item["item_id"], item.get("size"), tuple(sorted(item.get("modifiers") or [])))


def order_units(items: list[dict[str, Any]]) -> Counter:
    """Expand quantities so 2x margherita = two units. Lets us score partial credit."""
    c: Counter = Counter()
    for it in items:
        c[_unit_key(it)] += int(it.get("quantity", 1))
    return c


def item_prf(pred_items: list[dict], gold_items: list[dict]) -> dict[str, float]:
    pred, gold = order_units(pred_items), order_units(gold_items)
    tp = sum((pred & gold).values())
    p = tp / sum(pred.values()) if pred else (1.0 if not gold else 0.0)
    r = tp / sum(gold.values()) if gold else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def order_exact_match(pred: dict[str, Any], gold: dict[str, Any]) -> bool:
    if order_units(pred.get("items", [])) != order_units(gold.get("items", [])):
        return False
    for key in ("fulfilment", "customer_name", "status"):
        if key in gold and gold[key] is not None:
            if str(pred.get(key) or "").strip().lower() != str(gold[key]).strip().lower():
                return False
    return True


# ---------------------------------------------------------------- fine-tune actions


def parse_json_safely(text: str) -> Any | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _norm_action(a: dict[str, Any]) -> tuple:
    return (
        a.get("action"),
        a.get("item_id"),
        a.get("line"),
        a.get("quantity"),
        a.get("size"),
        tuple(sorted(a.get("modifiers") or [])),
    )


def actions_exact_match(pred: Any, gold: list[dict[str, Any]]) -> bool:
    if not isinstance(pred, list):
        return False
    try:
        return Counter(_norm_action(a) for a in pred) == Counter(_norm_action(a) for a in gold)
    except (AttributeError, TypeError):
        return False


# ---------------------------------------------------------------- speech


def menu_term_recall(reference: str, hypothesis: str, menu_terms: list[str]) -> float | None:
    """Of the menu words in what was said, how many did Whisper get right?"""
    ref, hyp = reference.lower(), hypothesis.lower()
    terms = [t for t in menu_terms if t.lower() in ref]
    if not terms:
        return None
    return sum(1 for t in terms if t.lower() in hyp) / len(terms)


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9àèéìòù' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- stats


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0
