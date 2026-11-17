"""MCP server exposing the ordering tools.

One server process = one conversation (one cart). Confirmed orders go to the shared SQLite DB,
so the kitchen display sees them no matter which client placed them.

Run it:
    uv run python -m mcp_server.menu_server                 # stdio (used by agent --mcp)
    npx @modelcontextprotocol/inspector uv run python -m mcp_server.menu_server   # click-test tools
"""

from __future__ import annotations

try:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # MCP Python SDK 2.x renamed FastMCP to MCPServer (same decorators)
    from mcp.server.mcpserver import MCPServer as FastMCP

from agent.config import path
from agent.db import OrderDB
from agent.order_state import Menu
from agent.tools import OrderTools

menu = Menu.load(path("menu"))
tools = OrderTools(menu, db=OrderDB(path("db")))

mcp = FastMCP("tabletalk-menu")


@mcp.tool()
def search_menu(query: str) -> str:
    """Find menu items matching what the customer said. Only needed when unsure which item they mean."""
    return tools.call("search_menu", {"query": query})


@mcp.tool()
def get_item(item_id: str) -> str:
    """Get sizes, prices, modifiers, allergens and availability for one item."""
    return tools.call("get_item", {"item_id": item_id})


@mcp.tool()
def add_item(item_id: str, quantity: int = 1, size: str | None = None,
             modifiers: list[str] | None = None, notes: str | None = None) -> str:
    """Add an item. item_id can be the id or the dish name the customer said. For several items call
    add_item once per item in the same response. Pizzas need a size: small, medium or large."""
    return tools.call("add_item", {"item_id": item_id, "quantity": quantity, "size": size,
                                   "modifiers": modifiers, "notes": notes})


@mcp.tool()
def update_item(line_no: int, quantity: int | None = None, size: str | None = None,
                modifiers: list[str] | None = None, item_id: str | None = None) -> str:
    """Change an order line: quantity, size, modifiers, or swap to another item_id."""
    return tools.call("update_item", {"line_no": line_no, "quantity": quantity, "size": size,
                                      "modifiers": modifiers, "item_id": item_id})


@mcp.tool()
def remove_item(item: str | None = None, line_no: int | None = None) -> str:
    """Remove an item from the order, by dish name (preferred) or line number."""
    return tools.call("remove_item", {"item": item, "line_no": line_no})


@mcp.tool()
def view_cart() -> str:
    """Show the current order with line numbers and the total."""
    return tools.call("view_cart", {})


@mcp.tool()
def check_allergens(item_id: str) -> str:
    """List the allergens of an item. Use for any allergy or dietary question."""
    return tools.call("check_allergens", {"item_id": item_id})


@mcp.tool()
def get_recommendation() -> str:
    """Get one suggestion to offer the customer (use at most once per order)."""
    return tools.call("get_recommendation", {})


@mcp.tool()
def confirm_order(fulfilment: str, customer_name: str) -> str:
    """Place the order (pickup or delivery). Only after reading the order back and the customer said yes."""
    return tools.call("confirm_order", {"fulfilment": fulfilment, "customer_name": customer_name})


@mcp.tool()
def cancel_order() -> str:
    """Cancel the whole order."""
    return tools.call("cancel_order", {})


@mcp.resource("menu://full")
def full_menu() -> str:
    """The whole menu as JSON."""
    return menu.model_dump_json(indent=2)


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
