import pytest

from agent.order_state import CartError


def test_menu_loads(menu):
    assert len(menu.items) >= 35
    ids = [i.id for i in menu.items]
    assert len(ids) == len(set(ids)), "item ids must be unique"


def test_add_sized_item_prices_correctly(cart):
    line = cart.add_item("pz_margherita", 2, "large")
    assert line.unit_price == 12.5
    assert cart.total == 25.0


def test_sized_item_without_size_asks(cart):
    with pytest.raises(CartError, match="Which size"):
        cart.add_item("pz_margherita", 1)


def test_drink_gets_default_size(cart):
    line = cart.add_item("dr_coke", 1)
    assert line.size == "medium" and line.unit_price == 3.0


def test_modifier_by_name_and_price(cart):
    line = cart.add_item("pz_margherita", 1, "small", ["extra mozzarella"])
    assert line.modifiers == ["extra_mozzarella"]
    assert line.unit_price == 10.0


def test_unknown_modifier_rejected(cart):
    with pytest.raises(CartError, match="isn't an option"):
        cart.add_item("pz_margherita", 1, "small", ["pineapple"])


def test_sold_out_rejected(cart):
    with pytest.raises(CartError, match="sold out"):
        cart.add_item("ds_tiramisu", 1)


def test_unknown_item_rejected(cart):
    with pytest.raises(CartError, match="not on our menu"):
        cart.add_item("burger", 1)


def test_quantity_limit(cart):
    with pytest.raises(CartError):
        cart.add_item("pz_margherita", 500, "large")


def test_update_quantity_and_swap(cart):
    cart.add_item("dr_coke", 1, "large")
    cart.update_item(1, quantity=2)
    assert cart.total == 7.0
    line = cart.update_item(1, item_id="dr_sprite")
    assert line.item_id == "dr_sprite" and line.size == "large" and line.quantity == 2
    assert cart.total == 7.0


def test_remove_and_line_numbers(cart):
    cart.add_item("st_fries", 1)
    cart.add_item("st_bruschetta", 1)
    cart.remove_item(1)
    line = cart.add_item("ds_cannoli", 1)
    assert line.line_no == 3  # line numbers are never reused
    assert cart.total == 13.0


def test_confirm_rules(cart):
    with pytest.raises(CartError, match="empty"):
        cart.confirm("pickup", "Marco")
    cart.add_item("ps_carbonara", 1)
    with pytest.raises(CartError, match="pickup or delivery"):
        cart.confirm("drone", "Marco")
    order = cart.confirm("pickup", "marco")
    assert order.status == "confirmed" and order.customer_name == "Marco"
    with pytest.raises(CartError, match="already confirmed"):
        cart.add_item("st_fries", 1)


def test_modifiers_understand_customer_words(menu):
    caesar, pasta, pizza = menu.get("st_caesar"), menu.get("ps_arrabbiata"), menu.get("pz_margherita")
    assert caesar.find_modifier("with chicken").id == "add_chicken"
    assert caesar.find_modifier("chicken").id == "add_chicken"
    assert pasta.find_modifier("gluten free").id == "gluten_free_pasta"
    assert pizza.find_modifier("no garlic") is None          # not an option on Margherita
    assert pizza.find_modifier("mozzarella").id == "extra_mozzarella"
