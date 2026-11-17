"""The eval harness itself must be correct, or the scores mean nothing."""

from agent.llm import LLMReply, ScriptedLLM, ToolCall
from evals.run_text_evals import run_conversation

CASE = {"id": "t", "persona": "quick", "goal": "1 Lasagna. Pickup. Name: Rosa.",
        "gold_order": {"items": [{"item_id": "ps_lasagna", "quantity": 1, "size": None, "modifiers": []}],
                       "fulfilment": "pickup", "customer_name": "Rosa"}}


def test_echo_is_removed_and_yes_before_goodbye_reaches_agent(menu):
    greeting = "Hi, thanks for calling Nonna's Kitchen! What can I get for you today?"
    customer = ScriptedLLM([
        LLMReply(greeting + " A lasagna please, pickup, name Rosa."),   # echoes the greeting first
        LLMReply("Yes, please place it. GOODBYE"),                       # yes + goodbye in one message
    ])
    agent = ScriptedLLM([
        LLMReply("", [ToolCall("add_item", {"item_id": "lasagna", "quantity": 1})]),
        LLMReply("One lasagna, €14.00, pickup for Rosa. Shall I place the order?"),
        LLMReply("", [ToolCall("confirm_order", {"fulfilment": "pickup", "customer_name": "Rosa"})]),
        LLMReply("Your order is placed, Rosa!"),
    ])
    r = run_conversation(CASE, agent, customer, menu, max_turns=6)
    assert r["transcript"][1]["customer"].startswith("A lasagna")
    assert r["completed"] and r["exact_match"]


def test_early_goodbye_does_not_end_the_call(menu):
    customer = ScriptedLLM([
        LLMReply("A lasagna, pickup, I'm Rosa. Yes place it. GOODBYE"),  # jumps ahead
        LLMReply("Yes, please place it."),
        LLMReply("GOODBYE"),
    ])
    agent = ScriptedLLM([
        LLMReply("", [ToolCall("add_item", {"item_id": "lasagna", "quantity": 1})]),
        LLMReply("One lasagna, €14.00, pickup for Rosa. Shall I place the order?"),
        LLMReply("", [ToolCall("confirm_order", {"fulfilment": "pickup", "customer_name": "Rosa"})]),
        LLMReply("Your order is placed, Rosa!"),
    ])
    r = run_conversation(CASE, agent, customer, menu, max_turns=6)
    assert r["completed"] and r["exact_match"]
