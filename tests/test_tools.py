import json

from agent.db import OrderDB
from agent.tools import TOOL_NAMES, OrderTools


def call(tools, name, **args):
    return json.loads(tools.call(name, args))


def test_every_schema_has_a_method(menu):
    tools = OrderTools(menu)
    for name in TOOL_NAMES:
        assert callable(getattr(tools, name))


def test_search_handles_aliases_and_typos(menu):
    tools = OrderTools(menu)
    assert call(tools, "search_menu", query="coke")["data"][0]["item_id"] == "dr_coke"
    assert call(tools, "search_menu", query="spright")["data"][0]["item_id"] == "dr_sprite"
    assert call(tools, "search_menu", query="four cheese pizza")["data"][0]["item_id"] == "pz_quattro_formaggi"
    assert call(tools, "search_menu", query="brushetta")["data"][0]["item_id"] == "st_bruschetta"


def test_search_flags_sold_out(menu):
    r = call(OrderTools(menu), "search_menu", query="tiramisu")
    assert "SOLD OUT" in r["message"]


def test_errors_come_back_as_messages_not_exceptions(menu):
    tools = OrderTools(menu)
    r = call(tools, "add_item", item_id="pz_margherita", quantity=1)
    assert r["ok"] is False and "size" in r["message"]
    r = call(tools, "add_item", wrong_arg=1)
    assert r["ok"] is False and "Bad arguments" in r["message"]
    assert call(tools, "not_a_tool")["ok"] is False


def test_full_order_flow_saves_to_db(menu, tmp_path):
    db = OrderDB(tmp_path / "t.db")
    confirmed = []
    tools = OrderTools(menu, db=db, on_confirm=lambda order, oid: confirmed.append(oid))
    call(tools, "add_item", item_id="pz_margherita", quantity=2, size="large", modifiers=["no basil"])
    call(tools, "add_item", item_id="dr_coke", quantity=1)
    call(tools, "update_item", line_no=2, item_id="dr_sprite")
    r = call(tools, "view_cart")
    assert r["data"]["total"] == 28.0
    r = call(tools, "confirm_order", fulfilment="pickup", customer_name="Marco")
    assert r["ok"] and confirmed == [1]
    saved = db.list_orders()
    assert saved[0]["customer_name"] == "Marco" and saved[0]["total"] == 28.0


def test_recommendation_only_once(menu):
    tools = OrderTools(menu)
    call(tools, "add_item", item_id="pz_margherita", quantity=1, size="small")
    first = call(tools, "get_recommendation")
    assert first["ok"] and "drink" in first["message"].lower()
    assert call(tools, "get_recommendation")["ok"] is False


def test_allergens_come_from_data(menu):
    r = call(OrderTools(menu), "check_allergens", item_id="ps_pesto")
    assert "nuts" in r["data"]["allergens"]
    assert "never promise" in r["message"]


def test_tools_accept_dish_names(menu):
    tools = OrderTools(menu)
    assert call(tools, "add_item", item_id="four cheese", quantity=1, size="small")["ok"]
    assert call(tools, "update_item", line_no=1, item_id="diavola")["data"]["items"][0]["item_id"] == "pz_diavola"
    assert call(tools, "check_allergens", item_id="pesto")["data"]["item_id"] == "ps_pesto"
    r = call(tools, "add_item", item_id="cheeseburger deluxe", quantity=1)
    assert r["ok"] is False and "search_menu" in r["message"]


def test_remove_by_name_never_removes_the_wrong_line(menu):
    tools = OrderTools(menu)
    call(tools, "add_item", item_id="garlic bread", quantity=1)
    call(tools, "add_item", item_id="diavola", quantity=1, size="medium")
    r = call(tools, "remove_item", item="fries")              # not in the order
    assert r["ok"] is False and len(r["data"]["items"]) == 2
    r = call(tools, "remove_item", item="garlic bread")
    assert [li["item_id"] for li in r["data"]["items"]] == ["pz_diavola"]
