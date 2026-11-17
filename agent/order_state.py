"""Menu, order and cart models.

Key design rule: the LLM never holds the order in its head and never computes prices.
The Cart below is the single source of truth, and the model can only change it through tools.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

Size = Literal["small", "medium", "large"]
Fulfilment = Literal["pickup", "delivery"]
MAX_QTY_PER_LINE = 20


class CartError(Exception):
    """A request the cart cannot fulfil. The message is written to be read back to the customer."""


# ---------------------------------------------------------------- menu


class Modifier(BaseModel):
    id: str
    name: str
    price: float = 0.0


class MenuItem(BaseModel):
    id: str
    name: str
    category: str
    description: str = ""
    aliases: list[str] = []
    price: float | None = None  # fixed price items
    sizes: dict[str, float] | None = None  # sized items
    default_size: str | None = None
    modifiers: list[Modifier] = []
    allergens: list[str] = []
    available: bool = True

    @property
    def has_sizes(self) -> bool:
        return bool(self.sizes)

    def unit_price(self, size: str | None) -> float:
        if self.sizes:
            if size is None:
                raise CartError(f"{self.name} comes in {', '.join(self.sizes)}. Which size would you like?")
            if size not in self.sizes:
                raise CartError(f"{self.name} comes in {', '.join(self.sizes)}, not {size}.")
            return self.sizes[size]
        if self.price is None:
            raise CartError(f"{self.name} has no price set.")
        return self.price

    def find_modifier(self, text: str) -> Modifier | None:
        """Match 'add chicken', 'with chicken', 'chicken', 'gluten free', 'gluten-free pasta'..."""
        key = _norm(text)
        if not key:
            return None
        for m in self.modifiers:  # 1) exact
            if key in {_norm(m.id), _norm(m.name)}:
                return m
        stripped = _norm(re.sub(r"\b(with|add|please|a|an|the|some)\b", " ", text.lower()))
        hits = [m for m in self.modifiers
                if stripped and len(stripped) >= 4 and (stripped in _norm(m.name) or _norm(m.name) in stripped)]
        if len(hits) == 1:  # 2) one clear partial match (never guess between two options)
            return hits[0]
        return None


class Menu(BaseModel):
    restaurant: str
    currency: str = "EUR"
    items: list[MenuItem]

    @classmethod
    def load(cls, path: str | Path) -> Menu:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(json.load(f))

    def get(self, item_id: str) -> MenuItem | None:
        return next((i for i in self.items if i.id == item_id), None)

    def all_prices(self) -> set[float]:
        prices: set[float] = set()
        for item in self.items:
            if item.price is not None:
                prices.add(round(item.price, 2))
            for p in (item.sizes or {}).values():
                prices.add(round(p, 2))
            for m in item.modifiers:
                prices.add(round(m.price, 2))
        return prices


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


# ---------------------------------------------------------------- order


class OrderItem(BaseModel):
    line_no: int
    item_id: str
    name: str
    quantity: int = Field(ge=1, le=MAX_QTY_PER_LINE)
    size: Size | None = None
    modifiers: list[str] = []  # modifier ids
    notes: str | None = None
    unit_price: float
    line_total: float

    def describe(self) -> str:
        size = f"{self.size} " if self.size else ""
        mods = f" ({', '.join(m.replace('_', ' ') for m in self.modifiers)})" if self.modifiers else ""
        return f"{self.quantity}x {size}{self.name}{mods}"


class Order(BaseModel):
    items: list[OrderItem] = []
    fulfilment: Fulfilment | None = None
    customer_name: str | None = None
    status: Literal["building", "confirmed", "cancelled"] = "building"
    total: float = 0.0


class Cart:
    """Holds one customer's order. Every change is validated against the menu."""

    def __init__(self, menu: Menu):
        self.menu = menu
        self.order = Order()

    # ---------- helpers
    def _item(self, item_id: str) -> MenuItem:
        item = self.menu.get(item_id)
        if item is None:
            raise CartError(f"'{item_id}' is not on our menu. Use search_menu to find the right item.")
        return item

    def _line(self, line_no: int) -> OrderItem:
        for line in self.order.items:
            if line.line_no == line_no:
                return line
        raise CartError(f"There is no line {line_no} in the order.")

    def _check_open(self) -> None:
        if self.order.status != "building":
            raise CartError(f"The order is already {self.order.status} and can't be changed.")

    def _price_line(self, item: MenuItem, size: str | None, modifiers: list[str]) -> float:
        unit = item.unit_price(size)
        for mod_id in modifiers:
            mod = item.find_modifier(mod_id)
            unit += mod.price if mod else 0.0
        return round(unit, 2)

    def _resolve_modifiers(self, item: MenuItem, modifiers: list[str] | None) -> list[str]:
        ids: list[str] = []
        for text in modifiers or []:
            mod = item.find_modifier(text)
            if mod is None:
                options = ", ".join(m.name for m in item.modifiers) or "none"
                raise CartError(f"NOT ADDED: '{text}' isn't an option for {item.name}. Its options are: {options}. "
                                "Call the tool again with one of these, or ask the customer.")
            if mod.id not in ids:
                ids.append(mod.id)
        return ids

    def _recalc(self) -> None:
        self.order.total = round(sum(line.line_total for line in self.order.items), 2)

    def _next_line_no(self) -> int:
        return max((line.line_no for line in self.order.items), default=0) + 1

    # ---------- actions
    def add_item(
        self,
        item_id: str,
        quantity: int = 1,
        size: str | None = None,
        modifiers: list[str] | None = None,
        notes: str | None = None,
    ) -> OrderItem:
        self._check_open()
        item = self._item(item_id)
        if not item.available:
            raise CartError(f"Sorry, {item.name} is sold out today.")
        if quantity < 1:
            raise CartError("Quantity must be at least 1.")
        if quantity > MAX_QTY_PER_LINE:
            raise CartError(f"For more than {MAX_QTY_PER_LINE} of one item please call the restaurant directly.")
        if size is None and item.has_sizes and item.default_size:
            size = item.default_size
        if size is not None and not item.has_sizes:
            size = None  # ignore sizes on fixed-price items
        mod_ids = self._resolve_modifiers(item, modifiers)
        unit = self._price_line(item, size, mod_ids)
        try:
            line = OrderItem(
                line_no=self._next_line_no(),
                item_id=item.id,
                name=item.name,
                quantity=quantity,
                size=size,  # type: ignore[arg-type]
                modifiers=mod_ids,
                notes=notes,
                unit_price=unit,
                line_total=round(unit * quantity, 2),
            )
        except ValidationError as e:
            raise CartError(f"Invalid item details: {e.errors()[0]['msg']}") from e
        self.order.items.append(line)
        self._recalc()
        return line

    def update_item(
        self,
        line_no: int,
        quantity: int | None = None,
        size: str | None = None,
        modifiers: list[str] | None = None,
        item_id: str | None = None,
        notes: str | None = None,
    ) -> OrderItem:
        """Change a line. Passing item_id swaps the product (e.g. 'make the coke a sprite')."""
        self._check_open()
        line = self._line(line_no)
        new_item = self._item(item_id) if item_id else self._item(line.item_id)
        if item_id and not new_item.available:
            raise CartError(f"Sorry, {new_item.name} is sold out today.")
        new_qty = line.quantity if quantity is None else quantity
        if new_qty < 1:
            raise CartError("Quantity must be at least 1. Use remove_item to delete a line.")
        if new_qty > MAX_QTY_PER_LINE:
            raise CartError(f"For more than {MAX_QTY_PER_LINE} of one item please call the restaurant directly.")
        if size is not None:
            new_size = size if new_item.has_sizes else None
        elif new_item.has_sizes:
            new_size = line.size if (line.size and line.size in (new_item.sizes or {})) else new_item.default_size
        else:
            new_size = None
        if modifiers is not None:
            new_mods = self._resolve_modifiers(new_item, modifiers)
        else:  # keep modifiers that still apply to the (possibly new) item
            new_mods = [m for m in line.modifiers if new_item.find_modifier(m)]
        unit = self._price_line(new_item, new_size, new_mods)
        updated = line.model_copy(
            update={
                "item_id": new_item.id,
                "name": new_item.name,
                "quantity": new_qty,
                "size": new_size,
                "modifiers": new_mods,
                "notes": notes if notes is not None else line.notes,
                "unit_price": unit,
                "line_total": round(unit * new_qty, 2),
            }
        )
        idx = self.order.items.index(line)
        self.order.items[idx] = updated
        self._recalc()
        return updated

    def remove_item(self, line_no: int) -> OrderItem:
        self._check_open()
        line = self._line(line_no)
        self.order.items.remove(line)
        self._recalc()
        return line

    def confirm(self, fulfilment: str, customer_name: str) -> Order:
        self._check_open()
        if not self.order.items:
            raise CartError("The order is empty. Add at least one item first.")
        if fulfilment not in ("pickup", "delivery"):
            raise CartError("Please choose pickup or delivery.")
        if not customer_name or not customer_name.strip():
            raise CartError("I need a name for the order.")
        self.order.fulfilment = fulfilment  # type: ignore[assignment]
        self.order.customer_name = customer_name.strip().title()
        self.order.status = "confirmed"
        return self.order

    def cancel(self) -> None:
        self.order = Order(status="cancelled")

    # ---------- views
    @property
    def total(self) -> float:
        return self.order.total

    def summary(self) -> str:
        if not self.order.items:
            return "empty"
        lines = [f"line {li.line_no}: {li.describe()} = €{li.line_total:.2f}" for li in self.order.items]
        return "; ".join(lines) + f". Total €{self.total:.2f}"

    def snapshot(self) -> dict:
        return self.order.model_dump()
