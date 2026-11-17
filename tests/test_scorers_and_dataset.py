import json

from evals.scorers import (
    actions_exact_match,
    item_prf,
    menu_term_recall,
    order_exact_match,
    parse_json_safely,
)
from finetune.make_dataset import build


def test_item_prf_partial_credit():
    gold = [{"item_id": "a", "quantity": 2, "size": "large", "modifiers": []}]
    pred = [{"item_id": "a", "quantity": 1, "size": "large", "modifiers": []}]
    m = item_prf(pred, gold)
    assert m["precision"] == 1.0 and m["recall"] == 0.5


def test_order_exact_match_checks_name_and_fulfilment():
    items = [{"item_id": "a", "quantity": 1, "size": None, "modifiers": ["x"]}]
    gold = {"items": items, "fulfilment": "pickup", "customer_name": "Marco"}
    assert order_exact_match({"items": items, "fulfilment": "pickup", "customer_name": "marco"}, gold)
    assert not order_exact_match({"items": items, "fulfilment": "delivery", "customer_name": "Marco"}, gold)


def test_action_matching_and_json_parsing():
    gold = [{"action": "remove", "line": 2}]
    assert actions_exact_match(parse_json_safely('Sure! [{"action":"remove","line":2}]'), gold)
    assert parse_json_safely("not json") is None


def test_menu_term_recall():
    assert menu_term_recall("two margherita and a tiramisu", "two margarita and a tiramisu",
                            ["margherita", "tiramisu"]) == 0.5


def test_dataset_examples_are_valid():
    rows = build(n=150, seed=1, n_paraphrase=0, model="none")
    assert len(rows) == 150
    for r in rows:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert isinstance(json.loads(r["messages"][2]["content"]), list)
