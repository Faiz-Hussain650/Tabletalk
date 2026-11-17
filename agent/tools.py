"""The ordering tools. One OrderTools object = one customer's conversation.

These same functions are:
  * exposed over MCP by mcp_server/menu_server.py, and
  * called in-process by the agent (LocalToolBackend) for the lowest latency.

Every tool returns a JSON string: {"ok": bool, "message": str, "data": ...}
`message` is short, plain English the model can read back to the customer.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from rapidfuzz import fuzz, process

from agent.db import OrderDB
from agent.order_state import Cart, CartError, Menu, Order

# JSON schemas in the format Ollama / OpenAI-style tool calling expects.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_menu",
            "description": "Find menu items matching what the customer said. Only needed when you are unsure "
            "which item they mean; add_item already understands dish names and nicknames.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What the customer asked for, e.g. 'coke' or 'four cheese pizza'"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item",
            "description": "Get sizes, prices, modifiers, allergens and availability for one item.",
            "parameters": {
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": "Add an item to the order. For several items, call add_item once per item in the SAME "
            "response. If a pizza has no size yet, ask the customer before adding it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "the item id, or simply the dish name the customer said"},
                    "quantity": {"type": "integer", "minimum": 1},
                    "size": {"type": "string", "enum": ["small", "medium", "large"]},
                    "modifiers": {"type": "array", "items": {"type": "string"},
                                  "description": "options in the customer's words, e.g. ['no basil'], ['add chicken'], ['gluten-free pasta']"},
                    "notes": {"type": "string"},
                },
                "required": ["item_id", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_item",
            "description": "Change an existing order line: quantity, size, modifiers, or swap to another item_id "
            "(e.g. 'make the coke a sprite'). Use view_cart to find line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line_no": {"type": "integer"},
                    "quantity": {"type": "integer", "minimum": 1},
                    "size": {"type": "string", "enum": ["small", "medium", "large"]},
                    "modifiers": {"type": "array", "items": {"type": "string"}},
                    "item_id": {"type": "string", "description": "new item id or dish name, to swap the item"},
                },
                "required": ["line_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_item",
            "description": "Remove an item from the order. Prefer passing the item name the customer said; "
            "line_no is optional.",
            "parameters": {"type": "object", "properties": {
                "item": {"type": "string", "description": "dish name, e.g. 'fries'"},
                "line_no": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_cart",
            "description": "Show the current order with line numbers and the total (computed by the system).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_allergens",
            "description": "List the allergens of an item. Use this for ANY allergy or dietary question.",
            "parameters": {
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommendation",
            "description": "Get one suggestion to offer the customer (use at most once per order).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_order",
            "description": "Place the order. Only call after you read the full order and total back AND the customer said yes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fulfilment": {"type": "string", "enum": ["pickup", "delivery"]},
                    "customer_name": {"type": "string"},
                },
                "required": ["fulfilment", "customer_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel the whole order when the customer asks to.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


def _result(ok: bool, message: str, data: Any = None) -> str:
    return json.dumps({"ok": ok, "message": message, "data": data}, ensure_ascii=False)


class OrderTools:
    def __init__(
        self,
        menu: Menu,
        db: OrderDB | None = None,
        on_confirm: Callable[[Order, int], None] | None = None,
    ):
        self.menu = menu
        self.cart = Cart(menu)
        self.db = db
        self.on_confirm = on_confirm
        self.upsell_done = False
        self.order_id: int | None = None
        # Build the fuzzy-search index once: every name and alias points to its item id.
        self._index: list[tuple[str, str]] = []
        for item in menu.items:
            self._index.append((item.name.lower(), item.id))
            for alias in item.aliases:
                self._index.append((alias.lower(), item.id))

    # ---------------------------------------------------------- name resolution
    def _resolve(self, ref: str | None) -> str | None:
        """Accept an item id OR a spoken name ('coke', 'four cheese') and return the item id.

        Letting the model pass names saves a whole search_menu round trip per item (~1-3 s each).
        """
        if ref is None:
            return None
        if self.menu.get(ref):
            return ref
        query = ref.lower().replace("_", " ").strip()
        best = process.extractOne(query, [k for k, _ in self._index], scorer=fuzz.WRatio)
        if best and best[1] >= 85:
            return self._index[best[2]][1]
        raise CartError(f"I'm not sure which item '{ref}' is. Use search_menu to find it.")

    # ---------------------------------------------------------- dispatch
    def call(self, name: str, args: dict[str, Any] | None = None) -> str:
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            return _result(False, f"Unknown tool {name}.")
        try:
            return fn(**(args or {}))
        except CartError as e:
            return _result(False, str(e), self.cart.snapshot())
        except TypeError as e:  # wrong / missing arguments from the model
            return _result(False, f"Bad arguments for {name}: {e}")

    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    # ---------------------------------------------------------- tools
    def search_menu(self, query: str) -> str:
        query = (query or "").lower().strip()
        if not query:
            return _result(False, "Empty search.")
        matches = process.extract(query, [k for k, _ in self._index], scorer=fuzz.WRatio, limit=10)
        results, seen = [], set()
        for _text, score, idx in matches:
            item_id = self._index[idx][1]
            if score < 60 or item_id in seen:
                continue
            seen.add(item_id)
            item = self.menu.get(item_id)
            assert item is not None
            results.append(
                {
                    "item_id": item.id,
                    "name": item.name,
                    "category": item.category,
                    "available": item.available,
                    "sizes": item.sizes,
                    "price": item.price,
                    "score": round(score),
                }
            )
            if len(results) == 3:
                break
        if not results:
            return _result(False, f"Nothing on the menu matches '{query}'.", [])
        best = results[0]
        msg = f"Best match: {best['name']} ({best['item_id']})"
        if not best["available"]:
            msg += " - SOLD OUT today"
        return _result(True, msg, results)

    def get_item(self, item_id: str) -> str:
        item = self.menu.get(self._resolve(item_id) or "")
        if item is None:
            return _result(False, f"No item with id {item_id}.")
        return _result(True, f"{item.name}: {item.description}", item.model_dump())

    def add_item(self, item_id: str, quantity: int = 1, size: str | None = None,
                 modifiers: list[str] | None = None, notes: str | None = None) -> str:
        line = self.cart.add_item(self._resolve(item_id) or "", int(quantity), size, modifiers, notes)
        return _result(True, f"Added {line.describe()}. Order total €{self.cart.total:.2f}.", self.cart.snapshot())

    def update_item(self, line_no: int, quantity: int | None = None, size: str | None = None,
                    modifiers: list[str] | None = None, item_id: str | None = None) -> str:
        line = self.cart.update_item(int(line_no), quantity, size, modifiers, self._resolve(item_id))
        return _result(True, f"Line {line.line_no} is now {line.describe()}. Order total €{self.cart.total:.2f}.",
                       self.cart.snapshot())

    def remove_item(self, line_no: int | None = None, item: str | None = None) -> str:
        if item:  # by name is safer than by number: the model can't remove the wrong line
            item_id = self._resolve(item)
            matches = [li for li in self.cart.order.items if li.item_id == item_id]
            if not matches:
                return _result(False, f"Nothing removed: {item} is not in the order. Current order: {self.cart.summary()}",
                               self.cart.snapshot())
            line_no = matches[-1].line_no
        if line_no is None:
            return _result(False, "Say which item to remove.")
        line = self.cart.remove_item(int(line_no))
        return _result(True, f"Removed {line.describe()}. Order total €{self.cart.total:.2f}.", self.cart.snapshot())

    def view_cart(self) -> str:
        return _result(True, f"Current order: {self.cart.summary()}", self.cart.snapshot())

    def check_allergens(self, item_id: str) -> str:
        item = self.menu.get(self._resolve(item_id) or "")
        if item is None:
            return _result(False, f"No item with id {item_id}.")
        listed = ", ".join(item.allergens) if item.allergens else "none of the listed allergens"
        return _result(
            True,
            f"{item.name} contains: {listed}. Remind the customer to tell staff about severe allergies; "
            "the kitchen handles all allergens, so never promise an item is safe.",
            {"item_id": item.id, "allergens": item.allergens},
        )

    def get_recommendation(self) -> str:
        if self.upsell_done:
            return _result(False, "Already suggested something. Do not upsell again.")
        self.upsell_done = True
        cats = {self.menu.get(li.item_id).category for li in self.cart.order.items}  # type: ignore[union-attr]
        if "drink" not in cats:
            return _result(True, "Suggest a drink: Coca-Cola, Sprite or sparkling water.", {"item_ids": ["dr_coke", "dr_sprite", "dr_water_sparkling"]})
        if "dessert" not in cats:
            dessert = next(i for i in self.menu.items if i.category == "dessert" and i.available)
            return _result(True, f"Suggest dessert: {dessert.name} (€{dessert.price:.2f}).", {"item_ids": [dessert.id]})
        return _result(True, "Nothing to suggest; the order looks complete.", {"item_ids": []})

    def confirm_order(self, fulfilment: str, customer_name: str) -> str:
        order = self.cart.confirm(fulfilment, customer_name)
        if self.db is not None:
            self.order_id = self.db.save_order(order)
        if self.on_confirm is not None:
            self.on_confirm(order, self.order_id or 0)
        ref = f" Order number {self.order_id}." if self.order_id else ""
        return _result(
            True,
            f"Order confirmed for {order.customer_name}, {order.fulfilment}, total €{order.total:.2f}.{ref}",
            self.cart.snapshot() | {"order_id": self.order_id},
        )

    def cancel_order(self) -> str:
        self.cart.cancel()
        return _result(True, "Order cancelled.", self.cart.snapshot())


class LocalToolBackend:
    """Calls OrderTools in-process (fastest). Same interface as agent.mcp_client.McpToolBackend."""

    def __init__(self, tools: OrderTools):
        self.tools = tools

    def schemas(self) -> list[dict[str, Any]]:
        return self.tools.schemas()

    def call(self, name: str, args: dict[str, Any] | None = None) -> str:
        return self.tools.call(name, args)

    def close(self) -> None:
        pass
