import pytest

from agent.config import path
from agent.order_state import Cart, Menu


@pytest.fixture(scope="session")
def menu() -> Menu:
    return Menu.load(path("menu"))


@pytest.fixture
def cart(menu: Menu) -> Cart:
    return Cart(menu)
