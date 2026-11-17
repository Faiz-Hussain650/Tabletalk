"""Checks that run in code, not just in the prompt.

Input:  prompt-injection / manipulation attempts.
Output: every price spoken must exist (menu price, line total or cart total),
        no free/discount promises, no allergen claims without a tool call this turn,
        no promises that something is "safe" for an allergy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent.order_state import Menu

INJECTION_PATTERNS = [
    r"ignore (all |any |your |the )?(previous |prior |above )?(instructions|rules|prompt)",
    r"system prompt",
    r"you are now",
    r"developer mode",
    r"pretend (you are|to be)",
    r"new instructions",
    r"jailbreak",
]
MANIPULATION_PATTERNS = [
    r"\bfor free\b",
    r"\bis free\b",
    r"\b(manager|owner|boss) (said|says|told)",
    r"\b\d{1,3} ?% off\b",
    r"\bdiscount\b",
    r"\bon the house\b",
    r"\bdon'?t charge\b",
]

PRICE_RE = re.compile(r"(?:€\s*(\d+(?:[.,]\d{1,2})?))|(?:(\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?\b|eur\b))", re.I)
FREE_RE = re.compile(r"(?<!-)\b(free|free of charge|no charge|on the house|\d+ ?% off|discounts?|discounted|complimentary)\b", re.I)
NEGATION_RE = re.compile(r"\b(can'?t|cannot|can not|not|unable|no|don'?t|won'?t|isn'?t|aren'?t|unfortunately|sorry)\b", re.I)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
ALLERGEN_WORDS = re.compile(r"\b(allerg\w*|gluten|nut|nuts|peanut\w*|dairy|lactose|milk|egg|eggs|celery|fish|shellfish|molluscs?|sulphites?|vegan|coeliac|celiac)\b", re.I)
SAFETY_PROMISE_RE = re.compile(r"\b(completely safe|totally safe|100% safe|guarantee\w*|is safe for|no risk|allergen[- ]free)\b", re.I)
REASONING_RE = re.compile(r"^\s*(okay|ok|alright|hmm|wait)[,.]?\s+(let'?s|let me|so|the user|i need)", re.I)
CLAIMS_PLACED_RE = re.compile(r"\b(order (is|has been|was) (placed|confirmed|on its way)|i'?ll place (the|your) order|i('ve| have) placed|placed your order|your order is (confirmed|on its way|being prepared))\b", re.I)
ALLERGEN_TOOLS = {"check_allergens", "get_item"}

INJECTION_REPLY = "I can only help with food orders for Nonna's Kitchen. What would you like to order?"
DISCOUNT_REPLY = ("I'm not able to change prices or add discounts, but I'm happy to help with your order. "
                  "What would you like?")


@dataclass
class Violation:
    code: str
    detail: str


@dataclass
class InputCheck:
    flagged: bool
    kind: str | None = None  # "injection" | "manipulation"
    canned_reply: str | None = None


def check_input(text: str) -> InputCheck:
    low = text.lower()
    if any(re.search(p, low) for p in INJECTION_PATTERNS):
        return InputCheck(True, "injection", INJECTION_REPLY)
    if any(re.search(p, low) for p in MANIPULATION_PATTERNS):
        # Don't block - the customer may still order - but tell the agent to refuse the price change.
        return InputCheck(True, "manipulation", DISCOUNT_REPLY)
    return InputCheck(False)


def _to_float(s: str) -> float:
    return round(float(s.replace(",", ".")), 2)


def allowed_prices(menu: Menu, cart_snapshot: dict[str, Any]) -> set[float]:
    prices = set(menu.all_prices())
    prices.add(round(cart_snapshot.get("total", 0.0), 2))
    for line in cart_snapshot.get("items", []):
        prices.add(round(line["line_total"], 2))
        prices.add(round(line["unit_price"], 2))
    return prices


def _scrub_menu_terms(text: str, menu: Menu) -> str:
    """Remove modifier names and item names so their words don't look like claims."""
    names = {m.name for item in menu.items for m in item.modifiers} | {i.name for i in menu.items}
    for name in sorted(names, key=len, reverse=True):
        pattern = re.escape(name).replace(r"\-", "[- ]").replace(r"\ ", "[- ]")
        text = re.sub(pattern, " ", text, flags=re.I)
    return text


def check_reply(reply: str, menu: Menu, cart_snapshot: dict[str, Any], tools_used_this_turn: set[str]) -> list[Violation]:
    violations: list[Violation] = []
    scrubbed = _scrub_menu_terms(reply, menu)  # "gluten-free base" or "oat milk" are options, not claims

    allowed = allowed_prices(menu, cart_snapshot)
    for m in PRICE_RE.finditer(reply):
        value = _to_float(m.group(1) or m.group(2))
        if value not in allowed:
            violations.append(Violation("unknown_price", f"€{value:.2f} is not a menu price or the current total"))

    for sentence in SENTENCE_SPLIT.split(scrubbed):
        # "I can't offer discounts" is fine; "That one is free" is not.
        if FREE_RE.search(sentence) and not NEGATION_RE.search(sentence):
            violations.append(Violation("discount_promise", "reply offers something free or discounted"))
            break

    if ALLERGEN_WORDS.search(scrubbed) and not (tools_used_this_turn & ALLERGEN_TOOLS):
        violations.append(Violation("allergen_without_tool", "talks about allergens without checking the data"))

    if len(reply.split()) > 60 or REASONING_RE.match(reply):
        # Phone replies are short. A long one is usually the model "thinking out loud".
        violations.append(Violation("too_long_or_reasoning", "reply is reasoning or far too long for a phone call"))

    if CLAIMS_PLACED_RE.search(reply) and cart_snapshot.get("status") != "confirmed":
        violations.append(Violation("claims_placed", "says the order is placed but confirm_order has not succeeded"))

    if SAFETY_PROMISE_RE.search(reply):
        violations.append(Violation("safety_promise", "promises an item is safe for an allergy"))

    return violations


def safe_fallback(cart_summary: str) -> str:
    if cart_summary == "empty":
        return "Sorry, let me start again. What would you like to order?"
    return f"Sorry, let me double-check. So far I have: {cart_summary}. Anything else?"
